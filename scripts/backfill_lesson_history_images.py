#!/usr/bin/env python3
"""Backfill lesson-plan image references from history into image_index.db."""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, Set

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.core.settings import resolve_path
from src.ingestion.storage.image_storage import ImageStorage

IMAGE_ROUTE_RE = re.compile(r"/lesson-plan-image/([A-Za-z0-9_\-]+)")


def extract_image_ids_from_history(db_path: Path) -> Set[str]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT lesson_content, conversation_state FROM lesson_history"
        ).fetchall()
    finally:
        conn.close()

    image_ids: Set[str] = set()
    for lesson_content, conversation_state in rows:
        image_ids.update(IMAGE_ROUTE_RE.findall(lesson_content or ""))
        image_ids.update(_extract_ids_from_json_blob(conversation_state or "{}"))
    return image_ids


def _extract_ids_from_json_blob(raw: str) -> Set[str]:
    try:
        data = json.loads(raw)
    except Exception:
        return set()

    found: Set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
            return
        if isinstance(value, list):
            for item in value:
                walk(item)
            return
        if isinstance(value, str):
            found.update(IMAGE_ROUTE_RE.findall(value))

    walk(data)
    return found


def find_image_file(images_root: Path, image_id: str) -> Path | None:
    matches = sorted(images_root.rglob(f"{image_id}.*"))
    for match in matches:
        if match.is_file():
            return match
    return None


def infer_collection_and_doc_hash(images_root: Path, path: Path) -> tuple[str | None, str | None]:
    try:
        rel = path.resolve().relative_to(images_root.resolve())
    except Exception:
        return None, None

    parts = rel.parts
    collection = parts[0] if len(parts) >= 2 else None
    doc_hash = parts[1] if len(parts) >= 3 else None
    return collection, doc_hash


def backfill(image_ids: Iterable[str], storage: ImageStorage, images_root: Path) -> dict[str, int]:
    stats = {"referenced": 0, "already_indexed": 0, "registered": 0, "missing_file": 0}
    for image_id in sorted(set(image_ids)):
        stats["referenced"] += 1
        if storage.get_image_path(image_id):
            stats["already_indexed"] += 1
            continue

        path = find_image_file(images_root, image_id)
        if path is None:
            stats["missing_file"] += 1
            continue

        collection, doc_hash = infer_collection_and_doc_hash(images_root, path)
        page_num = None
        match = re.search(r"_(\d+)_\d+$", image_id)
        if match:
            try:
                page_num = int(match.group(1))
            except ValueError:
                page_num = None

        storage.register_image(
            image_id=image_id,
            file_path=path,
            collection=collection,
            doc_hash=doc_hash,
            page_num=page_num,
        )
        stats["registered"] += 1

    return stats


def main() -> int:
    history_db = resolve_path("data/db/lesson_history.db")
    image_db = resolve_path("data/db/image_index.db")
    images_root = resolve_path("data/images")

    image_ids = extract_image_ids_from_history(history_db)
    storage = ImageStorage(db_path=str(image_db), images_root=str(images_root))
    try:
        stats = backfill(image_ids, storage, images_root)
    finally:
        storage.close()

    print(
        "[OK] Lesson history image backfill "
        f"referenced={stats['referenced']} "
        f"already_indexed={stats['already_indexed']} "
        f"registered={stats['registered']} "
        f"missing_file={stats['missing_file']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
