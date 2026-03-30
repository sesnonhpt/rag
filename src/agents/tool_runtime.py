"""Minimal tool runtime with schema validation, retry, and tracing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel, ValidationError


class ToolRuntimeError(RuntimeError):
    """Raised when tool execution fails after retries."""

    def __init__(self, message: str, *, code: str = "execution_error", tool_name: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code
        self.tool_name = tool_name


@dataclass
class ToolSpec:
    """Tool registration metadata."""

    request_model: Type[BaseModel]
    response_model: Type[BaseModel]
    handler: Callable[[BaseModel], BaseModel | Dict[str, Any] | Any]
    retry: int = 1
    timeout_seconds: float = 20.0


class ToolRuntime:
    """Small runtime to execute typed tools with retries and trace hooks."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, name: str, spec: ToolSpec) -> None:
        self._tools[name] = spec

    def invoke(
        self,
        name: str,
        payload: Dict[str, Any],
        trace: Optional[Any] = None,
    ) -> BaseModel:
        if name not in self._tools:
            raise ToolRuntimeError(f"Tool not registered: {name}", code="not_registered", tool_name=name)

        spec = self._tools[name]
        try:
            request_obj = spec.request_model.model_validate(payload)
        except ValidationError as exc:
            raise ToolRuntimeError(
                f"Validation failed for tool {name}: {exc}",
                code="validation_error",
                tool_name=name,
            ) from exc

        attempts = max(1, int(spec.retry) + 1)
        last_exc: Optional[Exception] = None
        start_at = time.time()

        for attempt in range(1, attempts + 1):
            attempt_start = time.time()
            try:
                elapsed_total = attempt_start - start_at
                if elapsed_total > spec.timeout_seconds:
                    raise TimeoutError(f"Tool {name} timed out after {elapsed_total:.2f}s")

                raw = spec.handler(request_obj)
                if isinstance(raw, spec.response_model):
                    response_obj = raw
                else:
                    response_obj = spec.response_model.model_validate(raw)

                if trace is not None:
                    trace.record_stage(
                        f"tool_{name}",
                        {"attempt": attempt, "status": "ok"},
                        elapsed_ms=(time.time() - attempt_start) * 1000,
                    )
                return response_obj
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if trace is not None:
                    trace.record_stage(
                        f"tool_{name}",
                        {"attempt": attempt, "status": "error", "error": str(exc)},
                        elapsed_ms=(time.time() - attempt_start) * 1000,
                    )
                if attempt >= attempts:
                    break

        final_code = "execution_error"
        if isinstance(last_exc, TimeoutError):
            final_code = "timeout_error"
        elif isinstance(last_exc, ValidationError):
            final_code = "validation_error"
        elif isinstance(last_exc, ToolRuntimeError):
            final_code = last_exc.code

        raise ToolRuntimeError(
            f"Tool {name} failed after {attempts} attempts: {last_exc}",
            code=final_code,
            tool_name=name,
        ) from last_exc
