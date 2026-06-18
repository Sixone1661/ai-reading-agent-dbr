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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS comparison_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL UNIQUE,
                student_id TEXT NOT NULL,
                week_node TEXT NOT NULL,
                literature_title TEXT NOT NULL,
                comparison_literature_title TEXT,
                research_question_comparison TEXT,
                theory_comparison TEXT,
                method_comparison TEXT,
                evidence_comparison TEXT,
                contribution_limit_comparison TEXT,
                synthesis_reflection TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS student_reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL UNIQUE,
                student_id TEXT NOT NULL,
                week_node TEXT NOT NULL,
                literature_title TEXT NOT NULL,
                reflection_stage TEXT,
                evidence_use_reflection TEXT,
                revised_understanding TEXT,
                ai_dependency_reflection TEXT,
                remaining_questions TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS student_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL UNIQUE,
                student_id TEXT NOT NULL,
                week_node TEXT NOT NULL,
                literature_title TEXT NOT NULL,
                usefulness INTEGER,
                ease_of_use INTEGER,
                critical_reading_support INTEGER,
                ai_dependency_awareness INTEGER,
                satisfaction INTEGER,
                helpful_questions TEXT,
                improvement_suggestions TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS teacher_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL UNIQUE,
                student_id TEXT NOT NULL,
                week_node TEXT NOT NULL,
                literature_title TEXT NOT NULL,
                teacher_name TEXT,
                research_question_score INTEGER,
                theory_score INTEGER,
                methodology_score INTEGER,
                evidence_score INTEGER,
                limitations_score INTEGER,
                transfer_score INTEGER,
                report_logic_score INTEGER,
                feedback_text TEXT,
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

def _upsert_by_session(table: str, values: dict) -> None:
    """Insert or update a one-row-per-session research artifact."""
    columns = list(values.keys())
    placeholders = ", ".join("?" for _ in columns)
    update_columns = [col for col in columns if col != "session_id"]
    assignments = ", ".join(f"{col} = excluded.{col}" for col in update_columns)
    query = f"""
        INSERT INTO {table} ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT(session_id) DO UPDATE SET
            {assignments},
            updated_at = CURRENT_TIMESTAMP
    """
    with get_connection() as conn:
        conn.execute(query, tuple(values[col] for col in columns))
        conn.commit()


def upsert_comparison_note(
    session_id: int,
    student_id: str,
    week_node: str,
    literature_title: str,
    comparison_literature_title: str,
    research_question_comparison: str,
    theory_comparison: str,
    method_comparison: str,
    evidence_comparison: str,
    contribution_limit_comparison: str,
    synthesis_reflection: str,
) -> None:
    """Save the T2 double-literature comparison artifact."""
    _upsert_by_session(
        "comparison_notes",
        {
            "session_id": session_id,
            "student_id": student_id,
            "week_node": week_node,
            "literature_title": literature_title,
            "comparison_literature_title": comparison_literature_title,
            "research_question_comparison": research_question_comparison,
            "theory_comparison": theory_comparison,
            "method_comparison": method_comparison,
            "evidence_comparison": evidence_comparison,
            "contribution_limit_comparison": contribution_limit_comparison,
            "synthesis_reflection": synthesis_reflection,
        },
    )


def get_comparison_note(session_id: int) -> sqlite3.Row | None:
    """Return the comparison note for a session, if present."""
    with get_connection() as conn:
        return conn.execute("SELECT * FROM comparison_notes WHERE session_id = ?", (session_id,)).fetchone()


def upsert_student_reflection(
    session_id: int,
    student_id: str,
    week_node: str,
    literature_title: str,
    reflection_stage: str,
    evidence_use_reflection: str,
    revised_understanding: str,
    ai_dependency_reflection: str,
    remaining_questions: str,
) -> None:
    """Save a student's metacognitive reflection text."""
    _upsert_by_session(
        "student_reflections",
        {
            "session_id": session_id,
            "student_id": student_id,
            "week_node": week_node,
            "literature_title": literature_title,
            "reflection_stage": reflection_stage,
            "evidence_use_reflection": evidence_use_reflection,
            "revised_understanding": revised_understanding,
            "ai_dependency_reflection": ai_dependency_reflection,
            "remaining_questions": remaining_questions,
        },
    )


def get_student_reflection(session_id: int) -> sqlite3.Row | None:
    """Return a student reflection, if present."""
    with get_connection() as conn:
        return conn.execute("SELECT * FROM student_reflections WHERE session_id = ?", (session_id,)).fetchone()


def upsert_student_feedback(
    session_id: int,
    student_id: str,
    week_node: str,
    literature_title: str,
    usefulness: int,
    ease_of_use: int,
    critical_reading_support: int,
    ai_dependency_awareness: int,
    satisfaction: int,
    helpful_questions: str,
    improvement_suggestions: str,
) -> None:
    """Save a short student experience questionnaire."""
    _upsert_by_session(
        "student_feedback",
        {
            "session_id": session_id,
            "student_id": student_id,
            "week_node": week_node,
            "literature_title": literature_title,
            "usefulness": usefulness,
            "ease_of_use": ease_of_use,
            "critical_reading_support": critical_reading_support,
            "ai_dependency_awareness": ai_dependency_awareness,
            "satisfaction": satisfaction,
            "helpful_questions": helpful_questions,
            "improvement_suggestions": improvement_suggestions,
        },
    )


def get_student_feedback(session_id: int) -> sqlite3.Row | None:
    """Return student feedback, if present."""
    with get_connection() as conn:
        return conn.execute("SELECT * FROM student_feedback WHERE session_id = ?", (session_id,)).fetchone()


def upsert_teacher_feedback(
    session_id: int,
    student_id: str,
    week_node: str,
    literature_title: str,
    teacher_name: str,
    research_question_score: int,
    theory_score: int,
    methodology_score: int,
    evidence_score: int,
    limitations_score: int,
    transfer_score: int,
    report_logic_score: int,
    feedback_text: str,
) -> None:
    """Save teacher rubric scores and feedback for one student session."""
    _upsert_by_session(
        "teacher_feedback",
        {
            "session_id": session_id,
            "student_id": student_id,
            "week_node": week_node,
            "literature_title": literature_title,
            "teacher_name": teacher_name,
            "research_question_score": research_question_score,
            "theory_score": theory_score,
            "methodology_score": methodology_score,
            "evidence_score": evidence_score,
            "limitations_score": limitations_score,
            "transfer_score": transfer_score,
            "report_logic_score": report_logic_score,
            "feedback_text": feedback_text,
        },
    )


def get_teacher_feedback(session_id: int) -> sqlite3.Row | None:
    """Return teacher feedback, if present."""
    with get_connection() as conn:
        return conn.execute("SELECT * FROM teacher_feedback WHERE session_id = ?", (session_id,)).fetchone()

