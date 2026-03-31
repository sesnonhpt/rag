"""SQLite-backed task storage for async lesson generation jobs."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional


class LessonTaskStorage:
    """Persist async lesson task state so polling survives process restarts."""

    def __init__(self, db_path: str = "data/db/lesson_tasks.db") -> None:
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
                CREATE TABLE IF NOT EXISTS lesson_tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress_stage TEXT,
                    result_json TEXT,
                    error_json TEXT,
                    created_at REAL NOT NULL,
                    finished_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_lesson_tasks_created_at
                ON lesson_tasks(created_at DESC)
                """
            )
            conn.commit()
        finally:
            conn.close()

    def upsert(
        self,
        *,
        task_id: str,
        status: str,
        progress_stage: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
        created_at: Optional[float] = None,
        finished_at: Optional[float] = None,
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT created_at FROM lesson_tasks WHERE task_id = ?",
                (task_id,),
            )
            existing = cursor.fetchone()
            effective_created_at = float(existing[0]) if existing and existing[0] is not None else float(created_at or time.time())
            conn.execute(
                """
                INSERT INTO lesson_tasks (
                    task_id, status, progress_stage, result_json, error_json, created_at, finished_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    status = excluded.status,
                    progress_stage = excluded.progress_stage,
                    result_json = excluded.result_json,
                    error_json = excluded.error_json,
                    created_at = excluded.created_at,
                    finished_at = excluded.finished_at
                """,
                (
                    task_id,
                    status,
                    progress_stage,
                    json.dumps(result, ensure_ascii=False) if result is not None else None,
                    json.dumps(error, ensure_ascii=False) if error is not None else None,
                    effective_created_at,
                    finished_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT task_id, status, progress_stage, result_json, error_json, created_at, finished_at
                FROM lesson_tasks
                WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        def _load(raw: Optional[str]) -> Optional[Dict[str, Any]]:
            if not raw:
                return None
            try:
                value = json.loads(raw)
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                return None

        return {
            "task_id": row["task_id"],
            "status": row["status"],
            "progress_stage": row["progress_stage"],
            "result": _load(row["result_json"]),
            "error": _load(row["error_json"]),
            "created_at": row["created_at"],
            "finished_at": row["finished_at"],
        }

    def cleanup(self, max_age_seconds: int = 3600) -> None:
        cutoff = time.time() - max_age_seconds
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "DELETE FROM lesson_tasks WHERE created_at < ?",
                (cutoff,),
            )
            conn.commit()
        finally:
            conn.close()
