"""Compatibility facade for API models, lifecycle, and legacy helper names."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Request

from app.core.app_runtime import (
    build_components as core_build_components,
    global_exception_handler,
    lifespan,
)
from app.core.lesson_content_helpers import (
    extract_image_path_from_src,
    extract_image_resources,
    find_image_file_by_id,
    integrate_images_into_markdown,
    is_result_relevant_to_topic,
    looks_like_lesson_refusal,
    prioritize_visual_results,
    resolve_docx_image_path,
    resolve_image_file_path,
    sanitize_source_path,
    to_review_report_response,
)
from app.core.paths import PROJECT_ROOT, STATIC_DIR
from app.core.prompt_builders import (
    build_chat_prompt,
    build_lesson_plan_prompt,
    build_lesson_plan_prompt_fallback,
)
from app.core.runtime_helpers import (
    build_api_error_detail,
    format_sse_event,
    is_fast_mode_enabled,
    is_planner_llm_enabled,
    resolve_llm_auth_for_model,
    resolve_template_type_from_category,
)
from app.schemas.api_models import (
    ChatRequest,
    LessonPlanRequest,
    Citation,
    LessonImageResource,
    LessonReviewReportResponse,
    ChatResponse,
    LessonPlanResponse,
    ExportDocxRequest,
    LessonHistoryResponse,
    LessonTemplateCategoryItem,
    LessonTemplateCategoriesResponse,
    HealthResponse,
)
from src.ingestion.storage.image_storage import ImageStorage

# Ensure project root is on sys.path
_ROOT = PROJECT_ROOT
sys.path.insert(0, str(_ROOT))

_STATIC_DIR = STATIC_DIR

# Keep a compact compatibility surface while new code uses app.main/routers/services.
__all__ = [
    "ChatRequest",
    "LessonPlanRequest",
    "Citation",
    "LessonImageResource",
    "LessonReviewReportResponse",
    "ChatResponse",
    "LessonPlanResponse",
    "ExportDocxRequest",
    "LessonHistoryResponse",
    "LessonTemplateCategoryItem",
    "LessonTemplateCategoriesResponse",
    "HealthResponse",
    "lifespan",
    "global_exception_handler",
    "_STATIC_DIR",
    "_build_components",
    "_resolve_template_type",
    "_resolve_llm_auth_for_model",
    "_is_fast_mode_enabled",
    "_is_planner_llm_enabled",
    "_build_api_error_detail",
    "_sanitize_source_path",
    "_resolve_image_file_path",
    "_find_image_file_by_id",
    "_extract_image_path_from_src",
    "_resolve_docx_image_path",
    "_extract_image_resources",
    "_prioritize_visual_results",
    "_is_result_relevant_to_topic",
    "_looks_like_lesson_refusal",
    "_to_review_report_response",
    "_integrate_images_into_markdown",
    "_build_prompt",
    "_build_lesson_plan_prompt",
    "_build_lesson_plan_prompt_fallback",
    "_generate_lesson_plan_internal",
    "_format_sse_event",
    "app",
]


def _resolve_template_type(req: LessonPlanRequest) -> Optional[str]:
    return resolve_template_type_from_category(req.template_category)


def _resolve_llm_auth_for_model(model_name: Optional[str], settings: Any) -> tuple[Optional[str], Optional[str]]:
    return resolve_llm_auth_for_model(model_name, settings)


def _is_fast_mode_enabled() -> bool:
    return is_fast_mode_enabled()


def _is_planner_llm_enabled() -> bool:
    return is_planner_llm_enabled()


def _build_api_error_detail(
    *,
    code: str,
    message: str,
    stage: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    return build_api_error_detail(code=code, message=message, stage=stage, trace_id=trace_id)


def _build_components(settings: Any, collection: str) -> tuple:
    return core_build_components(settings, collection)


def _sanitize_source_path(source_path: Any) -> str:
    return sanitize_source_path(source_path)


def _resolve_image_file_path(raw_path: Any) -> Path:
    return resolve_image_file_path(raw_path)


def _find_image_file_by_id(image_id: str) -> Optional[Path]:
    return find_image_file_by_id(image_id)


def _extract_image_path_from_src(src: str, image_storage: Any) -> Optional[Path]:
    return extract_image_path_from_src(src, image_storage)


def _resolve_docx_image_path(src: str, image_storage: Any) -> Optional[Path]:
    return resolve_docx_image_path(src, image_storage)


def _extract_image_resources(
    results: List[Any],
    image_storage: Optional[ImageStorage] = None,
    collection: Optional[str] = None,
    max_images: int = 6,
    topic: Optional[str] = None,
) -> List[LessonImageResource]:
    return extract_image_resources(
        results,
        image_storage=image_storage,
        collection=collection,
        max_images=max_images,
        topic=topic,
    )


def _prioritize_visual_results(results: List[Any], query_plan: Any) -> List[Any]:
    return prioritize_visual_results(results, query_plan)


def _is_result_relevant_to_topic(topic: str, result: Any) -> bool:
    return is_result_relevant_to_topic(topic, result)


def _looks_like_lesson_refusal(content: str) -> bool:
    return looks_like_lesson_refusal(content)


def _to_review_report_response(report: Any) -> Optional[LessonReviewReportResponse]:
    return to_review_report_response(report)


def _integrate_images_into_markdown(lesson_plan_content: str, image_resources: List[LessonImageResource]) -> str:
    return integrate_images_into_markdown(lesson_plan_content, image_resources)


def _build_prompt(question: str, contexts: List[Any]) -> List[Any]:
    return build_chat_prompt(question, contexts)


def _build_lesson_plan_prompt(
    topic: str,
    contexts: List[Any],
    include_background: bool,
    include_facts: bool,
    include_examples: bool,
) -> List[Any]:
    return build_lesson_plan_prompt(
        topic=topic,
        contexts=contexts,
        include_background=include_background,
        include_facts=include_facts,
        include_examples=include_examples,
    )


def _build_lesson_plan_prompt_fallback(req: LessonPlanRequest) -> List[Any]:
    return build_lesson_plan_prompt_fallback(req)


def _generate_lesson_plan_internal(
    req: LessonPlanRequest,
    request: Request,
    progress_callback: Optional[Any] = None,
) -> LessonPlanResponse:
    from app.services.lesson_service import generate_lesson_plan_internal

    return generate_lesson_plan_internal(req=req, request=request, progress_callback=progress_callback)


def _format_sse_event(event: str, payload: Dict[str, Any]) -> bytes:
    return format_sse_event(event, payload)


# Backward-compatibility entrypoint for `uvicorn app.chat_api:app`.
from app.main import app  # noqa: E402
