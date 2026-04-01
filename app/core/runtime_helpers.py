"""Shared runtime helpers for API/services."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from src.core.templates import resolve_template_type_by_category

_GEMINI_MODEL_PREFIXES = ("gemini-", "gemma-")
_GEMINI_GATEWAY_BASE_URL = os.environ.get(
    "GEMINI_GATEWAY_BASE_URL",
    "https://gemini-gateway.xn--7dvnlw2c.top/v1",
)
_GEMINI_GATEWAY_API_KEY = os.environ.get("GEMINI_GATEWAY_API_KEY")


def resolve_template_type_from_category(category: Optional[str]) -> Optional[str]:
    return resolve_template_type_by_category(category)


def resolve_llm_auth_for_model(model_name: Optional[str], settings: Any) -> tuple[Optional[str], Optional[str]]:
    model = str(model_name or "").strip().lower()
    if model.startswith(_GEMINI_MODEL_PREFIXES) and _GEMINI_GATEWAY_API_KEY:
        return _GEMINI_GATEWAY_API_KEY, _GEMINI_GATEWAY_BASE_URL
    return settings.llm.api_key, settings.llm.base_url


def is_fast_mode_enabled() -> bool:
    return str(os.environ.get("LESSON_FAST_MODE", "false")).strip().lower() in {"1", "true", "yes", "on"}


def is_planner_llm_enabled() -> bool:
    return str(os.environ.get("LESSON_PLANNER_USE_LLM", "false")).strip().lower() in {"1", "true", "yes", "on"}


def build_api_error_detail(
    *,
    code: str,
    message: str,
    stage: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"code": code, "message": message}
    if stage:
        payload["stage"] = stage
    if trace_id:
        payload["trace_id"] = trace_id
    return payload


def format_sse_event(event: str, payload: Dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
