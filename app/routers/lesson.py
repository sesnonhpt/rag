from __future__ import annotations

import asyncio
import contextlib
import os
import re
from io import BytesIO
from typing import Any, Dict, Optional
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.core.lesson_content_helpers import resolve_docx_image_path
from app.core.runtime_helpers import build_api_error_detail, format_sse_event
from app.schemas.api_models import (
    ExportDocxRequest,
    LessonHistoryResponse,
    LessonPlanRequest,
    LessonPlanResponse,
    LessonTemplateCategoriesResponse,
    LessonTemplateCategoryItem,
)
from app.services.docx_export_service import build_lesson_docx_bytes
from app.services.lesson_service import generate_lesson_plan_internal
from src.core.templates import get_template_categories_payload
from src.observability.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/lesson-template-categories", response_model=LessonTemplateCategoriesResponse)
async def get_lesson_template_categories():
    return LessonTemplateCategoriesResponse(
        templates=[LessonTemplateCategoryItem(**item) for item in get_template_categories_payload()]
    )


@router.get("/lesson-history", response_model=LessonHistoryResponse)
async def get_lesson_history(request: Request, session_id: Optional[str] = None, limit: int = 8):
    storage = request.app.state.history_storage
    safe_limit = max(1, min(limit, 20))
    records = storage.list_records(limit=safe_limit, session_id=session_id)
    return LessonHistoryResponse(records=records)


@router.delete("/lesson-history/{record_id}")
async def delete_lesson_history(record_id: int, request: Request):
    storage = request.app.state.history_storage
    storage.delete_record(record_id)
    return {"ok": True}


@router.post("/lesson-plan", response_model=LessonPlanResponse)
async def generate_lesson_plan(req: LessonPlanRequest, request: Request):
    lesson_timeout_sec = float(os.environ.get("LESSON_PLAN_TIMEOUT_SEC", "85"))
    try:
        response_payload = await asyncio.wait_for(
            asyncio.to_thread(
                generate_lesson_plan_internal,
                req,
                request,
            ),
            timeout=lesson_timeout_sec,
        )
    except asyncio.TimeoutError:
        logger.exception("Lesson orchestration timed out")
        raise HTTPException(
            status_code=504,
            detail=build_api_error_detail(
                code="LESSON_TIMEOUT",
                message=f"生成超时（>{int(lesson_timeout_sec)}s），请重试或切换更快模型",
                stage="lesson_orchestration",
            ),
        )
    except Exception as e:
        logger.exception("Lesson orchestration failed")
        raise HTTPException(
            status_code=500,
            detail=build_api_error_detail(
                code="LESSON_ORCHESTRATION_ERROR",
                message=f"教案编排失败: {str(e)}"[:280],
                stage="lesson_orchestration",
            ),
        ) from e
    return response_payload


@router.post("/lesson-plan/export-docx")
async def export_lesson_plan_docx(req: ExportDocxRequest, request: Request):
    try:
        filename = re.sub(r'[\\/:*?"<>|]+', "_", req.title).strip() or "教案"
        image_storage = getattr(request.app.state, "image_storage", None)
        docx_bytes = build_lesson_docx_bytes(
            content_html=req.content_html,
            resolve_image_path=lambda src: resolve_docx_image_path(src, image_storage),
        )
        ascii_filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._") or "lesson-plan"
        headers = {
            "Content-Disposition": (
                f'attachment; filename="{ascii_filename}.docx"; '
                f"filename*=UTF-8''{quote(filename, safe='')}.docx"
            ),
            "Content-Length": str(len(docx_bytes)),
        }
        logger.info(
            "Lesson DOCX export succeeded title=%r size=%s html_length=%s",
            filename,
            len(docx_bytes),
            len(req.content_html or ""),
        )
        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers=headers,
        )
    except Exception as e:
        logger.exception("Lesson DOCX export failed title=%r html_length=%s", req.title, len(req.content_html or ""))
        error_text = str(e).strip() or repr(e)
        raise HTTPException(
            status_code=500,
            detail=build_api_error_detail(
                code="LESSON_DOCX_EXPORT_ERROR",
                message=f"DOCX 导出失败: {error_text}"[:280],
                stage="lesson_docx_export",
            ),
        ) from e


@router.post("/lesson-plan/stream")
async def stream_lesson_plan(req: LessonPlanRequest, request: Request):
    lesson_timeout_sec = float(os.environ.get("LESSON_PLAN_TIMEOUT_SEC", "180"))

    async def event_stream():
        queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(stage: str, payload: Dict[str, Any]) -> None:
            logger.info("lesson_plan.stream_progress topic=%s stage=%s", req.topic, stage)
            loop.call_soon_threadsafe(
                queue.put_nowait,
                format_sse_event("progress", {"stage": stage, **payload}),
            )

        async def run_generation() -> None:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        generate_lesson_plan_internal,
                        req,
                        request,
                        emit,
                    ),
                    timeout=lesson_timeout_sec,
                )
                result_payload = result.model_dump() if hasattr(result, "model_dump") else result.dict()
                await queue.put(format_sse_event("result", result_payload))
            except asyncio.TimeoutError:
                await queue.put(
                    format_sse_event(
                        "error",
                        {
                            "code": "LESSON_TIMEOUT",
                            "message": f"生成超时（>{int(lesson_timeout_sec)}s），请重试或切换更快模型",
                            "stage": "lesson_orchestration",
                        },
                    )
                )
            except Exception as exc:
                logger.exception("Lesson streaming orchestration failed")
                await queue.put(
                    format_sse_event(
                        "error",
                        {
                            "code": "LESSON_ORCHESTRATION_ERROR",
                            "message": f"教案编排失败: {str(exc)}"[:280],
                            "stage": "lesson_orchestration",
                        },
                    )
                )
            finally:
                await queue.put(None)

        worker = asyncio.create_task(run_generation())
        yield format_sse_event(
            "progress",
            {
                "stage": "queued",
                "topic": req.topic,
                "template_category": req.template_category or "comprehensive",
            },
        )

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            if not worker.done():
                worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
