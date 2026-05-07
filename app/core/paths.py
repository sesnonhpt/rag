"""Shared filesystem paths for API modules."""

from __future__ import annotations

from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_ROOT.parent
STATIC_DIR = APP_ROOT / "static"
DATA_DIR = PROJECT_ROOT / "data"
TEMP_UPLOADS_DIR = DATA_DIR / "temp_uploads"
PROCESSING_HISTORY_FILE = DATA_DIR / "processing_history.json"
