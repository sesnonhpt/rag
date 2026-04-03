from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.observability.dashboard.services.trace_service import TraceService

router = APIRouter()


@router.get("/traces")
async def list_traces(
    trace_type: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
):
    svc = TraceService()
    return {
        "traces": svc.list_traces(trace_type=trace_type, limit=limit),
    }


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: str):
    svc = TraceService()
    trace = svc.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace
