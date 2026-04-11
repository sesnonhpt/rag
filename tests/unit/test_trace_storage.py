from __future__ import annotations

from pathlib import Path

from src.core.trace import TraceCollector, TraceContext
from src.core.trace.trace_storage import TraceStorage
from src.observability.dashboard.services.trace_service import TraceService


def test_trace_collector_persists_to_sqlite_and_trace_service_reads_it(tmp_path: Path) -> None:
    traces_jsonl = tmp_path / "traces.jsonl"
    traces_db = tmp_path / "traces.db"

    collector = TraceCollector(traces_path=traces_jsonl)
    collector._storage = TraceStorage(db_path=traces_db)

    trace = TraceContext(trace_type="lesson_plan")
    trace.metadata["topic"] = "牛顿第三定律"
    trace.record_stage("lesson_agent_complete", {"ok": True}, elapsed_ms=12.5)
    trace.finish()

    collector.collect(trace)

    stored = TraceStorage(db_path=traces_db).get_trace(trace.trace_id)
    assert stored is not None
    assert stored["trace_id"] == trace.trace_id
    assert stored["trace_type"] == "lesson_plan"
    assert stored["metadata"]["topic"] == "牛顿第三定律"

    service = TraceService(traces_path=traces_jsonl)
    service.storage = TraceStorage(db_path=traces_db)
    traces = service.list_traces(trace_type="lesson_plan")

    assert len(traces) == 1
    assert traces[0]["trace_id"] == trace.trace_id
