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
                    template_category TEXT,
                    template_label TEXT,
                    subject TEXT,
                    created_at TEXT NOT NULL,
                    conversation_state TEXT,
                    lesson_preview TEXT
                )
                """
            )
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

    def add_record(
        self,
        *,
        session_id: Optional[str],
        topic: str,
        template_category: Optional[str],
        template_label: Optional[str],
        subject: Optional[str],
        created_at: str,
        conversation_state: Optional[Dict[str, Any]],
        lesson_preview: Optional[str],
    ) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                INSERT INTO lesson_history (
                    session_id,
                    topic,
                    template_category,
                    template_label,
                    subject,
                    created_at,
                    conversation_state,
                    lesson_preview
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    topic,
                    template_category,
                    template_label,
                    subject,
                    created_at,
                    json.dumps(conversation_state or {}, ensure_ascii=False),
                    lesson_preview or "",
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
                    "templateCategory": row["template_category"],
                    "templateLabel": row["template_label"],
                    "subject": row["subject"],
                    "createdAt": row["created_at"],
                    "conversationState": parsed_state,
                    "summary": row["lesson_preview"] or "",
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
