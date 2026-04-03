"""SQLite-backed history storage for lesson generation records."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class LessonHistoryStorage:
    """Persist lightweight lesson history records."""

    def __init__(self, db_path: str = "data/db/lesson_history.db") -> None:
        self.db_path = db_path
        self._ensure_database()

    def _ensure_database(self) -> None:
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lesson_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    topic TEXT NOT NULL,
                    notes TEXT,
                    template_category TEXT,
                    template_label TEXT,
                    subject TEXT,
                    created_at TEXT NOT NULL,
                    conversation_state TEXT,
                    lesson_preview TEXT,
                    lesson_content TEXT,
                    planning_mode TEXT,
                    used_autonomous_fallback INTEGER DEFAULT 0
                )
                """
            )
            self._ensure_column(conn, "lesson_history", "planning_mode", "TEXT")
            self._ensure_column(conn, "lesson_history", "used_autonomous_fallback", "INTEGER DEFAULT 0")
            self._ensure_column(conn, "lesson_history", "lesson_content", "TEXT")
            self._ensure_column(conn, "lesson_history", "notes", "TEXT")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_lesson_history_session
                ON lesson_history(session_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_lesson_history_created_at
                ON lesson_history(created_at DESC)
                """
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_spec: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        existing = {row[1] for row in rows}
        if column_name in existing:
            return
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_spec}")

    def add_record(
        self,
        *,
        session_id: Optional[str],
        topic: str,
        notes: Optional[str],
        template_category: Optional[str],
        template_label: Optional[str],
        subject: Optional[str],
        created_at: str,
        conversation_state: Optional[Dict[str, Any]],
        lesson_preview: Optional[str],
        lesson_content: Optional[str],
        planning_mode: Optional[str],
        used_autonomous_fallback: bool,
    ) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                INSERT INTO lesson_history (
                    session_id,
                    topic,
                    notes,
                    template_category,
                    template_label,
                    subject,
                    created_at,
                    conversation_state,
                    lesson_preview,
                    lesson_content,
                    planning_mode,
                    used_autonomous_fallback
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    topic,
                    notes or "",
                    template_category,
                    template_label,
                    subject,
                    created_at,
                    json.dumps(conversation_state or {}, ensure_ascii=False),
                    lesson_preview or "",
                    lesson_content or "",
                    planning_mode,
                    1 if used_autonomous_fallback else 0,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)
        finally:
            conn.close()

    def list_records(self, limit: int = 8, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            if session_id:
                rows = conn.execute(
                    """
                    SELECT * FROM lesson_history
                    WHERE session_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM lesson_history
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        finally:
            conn.close()

        records: List[Dict[str, Any]] = []
        for row in rows:
            conversation_state = row["conversation_state"] or "{}"
            try:
                parsed_state = json.loads(conversation_state)
            except json.JSONDecodeError:
                parsed_state = None
            records.append(
                {
                    "id": row["id"],
                    "topic": row["topic"],
                    "notes": row["notes"] or "",
                    "templateCategory": row["template_category"],
                    "templateLabel": row["template_label"],
                    "subject": row["subject"],
                    "createdAt": row["created_at"],
                    "conversationState": parsed_state,
                    "summary": row["lesson_preview"] or "",
                    "lessonContent": row["lesson_content"] or "",
                    "planningMode": row["planning_mode"] or "context_first",
                    "usedAutonomousFallback": bool(row["used_autonomous_fallback"] or 0),
                }
            )
        return records

    def delete_record(self, record_id: int) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM lesson_history WHERE id = ?", (record_id,))
            conn.commit()
        finally:
            conn.close()
