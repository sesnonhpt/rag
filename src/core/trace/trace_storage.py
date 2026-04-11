"""SQLite-backed persistent storage for traces."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import resolve_path


DEFAULT_TRACE_DB_PATH = resolve_path("data/db/traces.db")


class TraceStorage:
    """Persist traces in SQLite so they survive service restarts."""

    def __init__(self, db_path: str | Path = DEFAULT_TRACE_DB_PATH) -> None:
        self.db_path = str(db_path)
        self._ensure_database()

    def _ensure_database(self) -> None:
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    trace_type TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    total_elapsed_ms REAL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_traces_type_started
                ON traces(trace_type, started_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_traces_started
                ON traces(started_at DESC)
                """
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_trace(self, trace: Dict[str, Any]) -> None:
        trace_id = str(trace.get("trace_id") or "").strip()
        if not trace_id:
            raise ValueError("trace_id is required")

        payload_json = json.dumps(trace, ensure_ascii=False)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO traces (
                    trace_id,
                    trace_type,
                    started_at,
                    finished_at,
                    total_elapsed_ms,
                    payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id) DO UPDATE SET
                    trace_type = excluded.trace_type,
                    started_at = excluded.started_at,
                    finished_at = excluded.finished_at,
                    total_elapsed_ms = excluded.total_elapsed_ms,
                    payload_json = excluded.payload_json
                """,
                (
                    trace_id,
                    str(trace.get("trace_type") or ""),
                    str(trace.get("started_at") or ""),
                    trace.get("finished_at"),
                    trace.get("total_elapsed_ms"),
                    payload_json,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def list_traces(self, trace_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            if trace_type:
                rows = conn.execute(
                    """
                    SELECT payload_json
                    FROM traces
                    WHERE trace_type = ?
                    ORDER BY started_at DESC, trace_id DESC
                    LIMIT ?
                    """,
                    (trace_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT payload_json
                    FROM traces
                    ORDER BY started_at DESC, trace_id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        finally:
            conn.close()

        traces: List[Dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                traces.append(payload)
        return traces

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT payload_json FROM traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None
        try:
            payload = json.loads(row[0] or "{}")
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def count_traces(self, trace_type: Optional[str] = None) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            if trace_type:
                row = conn.execute(
                    "SELECT COUNT(*) FROM traces WHERE trace_type = ?",
                    (trace_type,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM traces").fetchone()
        finally:
            conn.close()
        return int(row[0] or 0) if row else 0

    def clear(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM traces")
            conn.commit()
        finally:
            conn.close()
