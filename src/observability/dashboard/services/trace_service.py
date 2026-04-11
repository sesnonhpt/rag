"""TraceService – read and parse traces from logs/traces.jsonl.

Provides a typed, filterable interface over the raw JSONL trace log.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from src.core.settings import resolve_path
from src.core.trace.trace_storage import TraceStorage

logger = logging.getLogger(__name__)

# Default path to the traces file (absolute, CWD-independent)
DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")


class TraceService:
    """Read-only service for querying recorded traces.

    Args:
        traces_path: Path to the JSONL file.  Defaults to
            ``logs/traces.jsonl``.
    """

    def __init__(self, traces_path: Optional[str | Path] = None) -> None:
        self.traces_path = Path(traces_path) if traces_path else DEFAULT_TRACES_PATH
        self.api_base_url = str(os.environ.get("API_BASE_URL", "") or "").strip().rstrip("/")
        self.storage = TraceStorage()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_traces(
        self,
        trace_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return traces in reverse-chronological order.

        Args:
            trace_type: Filter by ``trace_type`` field (e.g.
                ``"ingestion"`` or ``"query"``).  ``None`` = all.
            limit: Maximum number of traces to return.

        Returns:
            List of trace dicts (newest first).
        """
        traces = self._load_all()

        if trace_type:
            traces = [t for t in traces if t.get("trace_type") == trace_type]

        # Newest first
        traces.sort(key=lambda t: t.get("started_at", ""), reverse=True)

        return traces[:limit]

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single trace by its ``trace_id``.

        Returns:
            Trace dict, or ``None`` if not found.
        """
        for t in self._load_all():
            if t.get("trace_id") == trace_id:
                return t
        return None

    def get_stage_timings(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract stage timings from a trace.

        Returns:
            List of dicts with keys: stage_name, elapsed_ms, data.
            Ordered by appearance.
        """
        stages = trace.get("stages", [])
        timings: List[Dict[str, Any]] = []
        for idx, s in enumerate(stages):
            # The raw stage dict has: stage, timestamp, data (dict), elapsed_ms
            # Extract the inner 'data' dict directly rather than flattening
            stage_data = s.get("data", {})
            if not isinstance(stage_data, dict):
                stage_data = {}
            elapsed_ms = s.get("elapsed_ms")
            if elapsed_ms is None:
                elapsed_ms = self._infer_elapsed_from_next_stage(stages, idx)
            timings.append(
                {
                    "stage_name": s.get("stage"),
                    "elapsed_ms": elapsed_ms,
                    "data": stage_data,
                }
            )
        return timings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all(self) -> List[Dict[str, Any]]:
        """Parse every line in the JSONL file.

        Silently skips malformed lines.
        """
        remote_traces = self._load_all_from_api()
        if remote_traces is not None:
            return remote_traces

        sqlite_traces = self.storage.list_traces(limit=1000)
        if sqlite_traces:
            return sqlite_traces

        if not self.traces_path.exists():
            return []

        traces: List[Dict[str, Any]] = []
        with self.traces_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed trace line: %s", line[:80])
        return traces

    def _load_all_from_api(self) -> Optional[List[Dict[str, Any]]]:
        if not self.api_base_url:
            return None

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.api_base_url}/traces", params={"limit": 500})
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            logger.warning("TraceService API fallback failed: %r", exc)
            return None

        traces = payload.get("traces", [])
        if not isinstance(traces, list):
            return []
        return [item for item in traces if isinstance(item, dict)]

    @staticmethod
    def _parse_iso_ts(value: Any) -> Optional[datetime]:
        if not value:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except Exception:
            return None

    def _infer_elapsed_from_next_stage(self, stages: List[Dict[str, Any]], idx: int) -> Optional[float]:
        if idx < 0 or idx >= len(stages) - 1:
            return None
        cur = self._parse_iso_ts(stages[idx].get("timestamp"))
        nxt = self._parse_iso_ts(stages[idx + 1].get("timestamp"))
        if cur is None or nxt is None:
            return None
        delta_ms = (nxt - cur).total_seconds() * 1000.0
        if delta_ms < 0:
            return None
        return round(delta_ms, 2)
