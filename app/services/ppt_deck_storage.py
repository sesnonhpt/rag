"""SQLite-backed storage for editable PPT decks."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class PptDeckStorage:
    """Persist editable PPT decks as JSON payloads."""

    def __init__(self, db_path: str = "data/db/ppt_decks.db") -> None:
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
                CREATE TABLE IF NOT EXISTS ppt_decks (
                    deck_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    topic TEXT,
                    template_category TEXT,
                    slide_count INTEGER DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ppt_decks_updated_at
                ON ppt_decks(updated_at DESC)
                """
            )
            conn.commit()
        finally:
            conn.close()

    def save_deck(self, deck: Dict[str, Any]) -> str:
        deck_id = str(deck.get("deck_id") or "").strip()
        if not deck_id:
            raise ValueError("deck_id is required")

        title = str(deck.get("title") or "").strip()
        if not title:
            raise ValueError("title is required")

        slides = deck.get("slides") or []
        created_at = str(deck.get("created_at") or "")
        updated_at = str(deck.get("updated_at") or "")

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO ppt_decks (
                    deck_id,
                    title,
                    topic,
                    template_category,
                    slide_count,
                    payload_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(deck_id) DO UPDATE SET
                    title = excluded.title,
                    topic = excluded.topic,
                    template_category = excluded.template_category,
                    slide_count = excluded.slide_count,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    deck_id,
                    title,
                    deck.get("topic"),
                    deck.get("template_category"),
                    len(slides) if isinstance(slides, list) else 0,
                    json.dumps(deck, ensure_ascii=False),
                    created_at,
                    updated_at,
                ),
            )
            conn.commit()
            return deck_id
        finally:
            conn.close()

    def get_deck(self, deck_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT payload_json FROM ppt_decks WHERE deck_id = ?",
                (deck_id,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def list_decks(self, limit: int = 20) -> List[Dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 100))
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT deck_id, title, topic, template_category, slide_count, created_at, updated_at
                FROM ppt_decks
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        finally:
            conn.close()

        return [
            {
                "deck_id": row["deck_id"],
                "title": row["title"],
                "topic": row["topic"],
                "template_category": row["template_category"] or "ppt",
                "slide_count": int(row["slide_count"] or 0),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def delete_deck(self, deck_id: str) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM ppt_decks WHERE deck_id = ?", (deck_id,))
            conn.commit()
        finally:
            conn.close()
