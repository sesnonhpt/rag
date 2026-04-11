"""Experimental image generation service for the lesson-plan UI."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import http.client
import json
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request
import uuid

from app.core.paths import STATIC_DIR
from src.core.settings import DEFAULT_SETTINGS_PATH, load_settings
from src.observability.logger import get_logger

logger = get_logger(__name__)

_STYLE_SUFFIXES = {
    "diagram_clean": "输出简洁、清晰、扁平化的中文教学示意图，强调结构和标签，可读性优先。",
    "education_illustration": "输出适合课堂使用的教学插画风格图片，表达友好、不过度写实。",
    "minimal_infographic": "输出极简信息图风格，层次清楚，留白充足，适合课件插图。",
}


class ImageGenerationError(RuntimeError):
    """Raised when image generation fails."""


@dataclass(frozen=True)
class ImageGenerationConfig:
    enabled: bool
    model: Optional[str]
    api_key: Optional[str]
    base_url: Optional[str]
    output_dir: Path


@lru_cache(maxsize=1)
def _load_project_image_defaults() -> Dict[str, Optional[str]]:
    try:
        settings = load_settings(str(DEFAULT_SETTINGS_PATH))
    except Exception:
        return {
            "model": None,
            "api_key": None,
            "base_url": None,
        }

    vision_llm = getattr(settings, "vision_llm", None)
    llm = getattr(settings, "llm", None)
    return {
        "model": None,
        "api_key": getattr(vision_llm, "api_key", None) or getattr(llm, "api_key", None),
        "base_url": getattr(vision_llm, "base_url", None) or getattr(llm, "base_url", None),
    }


def get_image_generation_config() -> ImageGenerationConfig:
    project_defaults = _load_project_image_defaults()
    raw_enabled = os.environ.get("IMAGE_GENERATION_ENABLED")
    if raw_enabled is None:
        enabled = True
    else:
        enabled = raw_enabled.strip().lower() not in {"0", "false", "off", "no"}

    model = (
        os.environ.get("IMAGE_GENERATION_MODEL")
        or os.environ.get("OPENAI_IMAGE_MODEL")
        # Default to mini for the experiment module: quality is sufficient
        # for current teaching diagrams while latency/cost are noticeably lower.
        # Keep gpt-image-1.5 as the high-quality fallback when we explicitly
        # want better visual fidelity and can accept higher cost.
        or project_defaults.get("model")
        or "gpt-image-1-mini"
    )
    api_key = (
        os.environ.get("IMAGE_GENERATION_API_KEY")
        or os.environ.get("OPENAI_IMAGE_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or project_defaults.get("api_key")
    )
    base_url = (
        os.environ.get("IMAGE_GENERATION_BASE_URL")
        or os.environ.get("OPENAI_IMAGE_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or project_defaults.get("base_url")
        or "https://api.openai.com/v1"
    )
    output_dir = STATIC_DIR / "generated-images"
    return ImageGenerationConfig(
        enabled=enabled and bool(model and api_key and base_url),
        model=model,
        api_key=api_key,
        base_url=base_url,
        output_dir=output_dir,
    )


class ExperimentalImageGenerationService:
    """Minimal OpenAI-compatible image generation integration."""

    def __init__(self, config: Optional[ImageGenerationConfig] = None) -> None:
        self.config = config or get_image_generation_config()

    def generate_image(
        self,
        *,
        prompt: str,
        style: str,
        topic: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.config.enabled:
            raise ImageGenerationError(
                "图片生成功能未配置。请设置 IMAGE_GENERATION_API_KEY / IMAGE_GENERATION_BASE_URL / IMAGE_GENERATION_MODEL。"
            )

        clean_prompt = " ".join(str(prompt or "").split())
        if not clean_prompt:
            raise ImageGenerationError("提示词不能为空")

        final_prompt = self._build_prompt(prompt=clean_prompt, style=style, topic=topic)
        payload = {
            "model": self.config.model,
            "prompt": final_prompt,
            "size": "1024x1024",
            "response_format": "b64_json",
        }

        try:
            response_payload = self._request_image_payload(payload)
        except urllib_error.HTTPError as exc:
            detail = self._extract_error_body(exc)
            raise ImageGenerationError(detail or f"图片生成请求失败（HTTP {exc.code}）") from exc
        except urllib_error.URLError as exc:
            raise ImageGenerationError(f"图片生成网络异常：{exc.reason}") from exc
        except http.client.RemoteDisconnected as exc:
            raise ImageGenerationError("图片网关已连接但未返回有效响应，可能是不支持当前图片模型或图片生成接口。") from exc

        image_bytes = self._extract_image_bytes(response_payload)
        saved = self._save_image(image_bytes)
        logger.info(
            "experimental_image.generated model=%s topic=%s style=%s file=%s",
            self.config.model,
            topic or "",
            style,
            saved["filename"],
        )
        return {
            "image_url": saved["url"],
            "image_path": str(saved["path"]),
            "filename": saved["filename"],
            "model": self.config.model,
            "style": style,
            "prompt": clean_prompt,
            "topic": topic or "",
        }

    def _build_prompt(self, *, prompt: str, style: str, topic: Optional[str]) -> str:
        style_suffix = _STYLE_SUFFIXES.get(style, _STYLE_SUFFIXES["diagram_clean"])
        topic_text = f"主题：{topic}。" if topic else ""
        guardrail = (
            "用于教学场景。避免水印、品牌标识、敏感人物肖像和过度写实照片感。"
            "优先表达概念关系、流程、结构和课堂可解释性。"
        )
        return f"{topic_text}{prompt} {style_suffix} {guardrail}".strip()

    def _images_endpoint(self) -> str:
        base_url = str(self.config.base_url or "").rstrip("/")
        if base_url.endswith("/images/generations"):
            return base_url
        return f"{base_url}/images/generations"

    def _post_json(self, url: str, payload: Dict[str, Any], *, api_key: str) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib_request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))

    def _request_image_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        endpoint = self._images_endpoint()
        api_key = self.config.api_key or ""
        try:
            return self._post_json(endpoint, payload, api_key=api_key)
        except urllib_error.HTTPError as exc:
            detail = self._extract_error_body(exc)
            if "Unknown parameter: 'response_format'" not in detail:
                raise

            fallback_payload = dict(payload)
            fallback_payload.pop("response_format", None)
            return self._post_json(endpoint, fallback_payload, api_key=api_key)

    def _extract_error_body(self, exc: urllib_error.HTTPError) -> str:
        try:
            raw = exc.read().decode("utf-8")
            payload = json.loads(raw)
            error = payload.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
            if isinstance(payload.get("detail"), str):
                return str(payload["detail"])
            return raw[:300]
        except Exception:
            return ""

    def _extract_image_bytes(self, payload: Dict[str, Any]) -> bytes:
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise ImageGenerationError("图片生成返回为空")

        first = data[0]
        if not isinstance(first, dict):
            raise ImageGenerationError("图片生成返回格式异常")

        b64_value = first.get("b64_json")
        if isinstance(b64_value, str) and b64_value.strip():
            try:
                return base64.b64decode(b64_value)
            except Exception as exc:
                raise ImageGenerationError("图片内容解码失败") from exc

        image_url = first.get("url")
        if isinstance(image_url, str) and image_url.strip():
            try:
                with urllib_request.urlopen(image_url, timeout=120) as response:
                    return response.read()
            except Exception as exc:
                raise ImageGenerationError("远程图片下载失败") from exc

        raise ImageGenerationError("图片生成响应中未包含可用图片数据")

    def _save_image(self, image_bytes: bytes) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        output_dir = self.config.output_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"exp-{uuid.uuid4().hex[:12]}.png"
        file_path = output_dir / filename
        file_path.write_bytes(image_bytes)

        try:
            relative_parts = file_path.relative_to(STATIC_DIR).parts
        except ValueError:
            relative_parts = ("generated-images", timestamp, filename)
        return {
            "path": file_path,
            "filename": filename,
            "url": "/static/" + "/".join(relative_parts),
        }
