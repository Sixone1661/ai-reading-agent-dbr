"""Export helpers for teacher-side data review."""

from __future__ import annotations

import pandas as pd

from database import get_connection


def get_messages_dataframe() -> pd.DataFrame:
    """Return joined conversation logs as a DataFrame."""
    query = """
        SELECT
            m.id AS message_id,
            m.session_id,
            s.student_id,
            s.week_node,
            s.literature_title,
            s.task_type,
            m.role,
            m.content,
            m.created_at
        FROM messages m
        JOIN sessions s ON s.id = m.session_id
        ORDER BY m.created_at, m.id
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)


def get_reading_notes_dataframe() -> pd.DataFrame:
    """Return reading notes as a DataFrame."""
    query = """
        SELECT
            rn.id AS note_id,
            rn.session_id,
            rn.student_id,
            rn.week_node,
            rn.literature_title,
            rn.research_question,
            rn.theoretical_framework,
            rn.methodology,
            rn.evidence_chain,
            rn.limitations_contributions,
            rn.transfer_reflection,
            rn.open_questions,
            rn.created_at,
            rn.updated_at
        FROM reading_notes rn
        ORDER BY rn.updated_at DESC, rn.id DESC
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Encode a DataFrame as UTF-8 with BOM so Chinese text opens well in Excel."""
    return df.to_csv(index=False).encode("utf-8-sig")

def get_comparison_notes_dataframe() -> pd.DataFrame:
    """Return T2 double-literature comparison artifacts as a DataFrame."""
    query = """
        SELECT *
        FROM comparison_notes
        ORDER BY updated_at DESC, id DESC
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)


def get_student_reflections_dataframe() -> pd.DataFrame:
    """Return student metacognitive reflections as a DataFrame."""
    query = """
        SELECT *
        FROM student_reflections
        ORDER BY updated_at DESC, id DESC
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)


def get_student_feedback_dataframe() -> pd.DataFrame:
    """Return student questionnaire feedback as a DataFrame."""
    query = """
        SELECT *
        FROM student_feedback
        ORDER BY updated_at DESC, id DESC
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)


def get_teacher_feedback_dataframe() -> pd.DataFrame:
    """Return teacher rubric scores and feedback as a DataFrame."""
    query = """
        SELECT *
        FROM teacher_feedback
        ORDER BY updated_at DESC, id DESC
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)

