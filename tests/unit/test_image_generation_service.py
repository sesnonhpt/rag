from __future__ import annotations

import base64
import json
from pathlib import Path

from app.services.image_generation_service import (
    ExperimentalImageGenerationService,
    ImageGenerationConfig,
    ImageGenerationError,
    _load_project_image_defaults,
    get_image_generation_config,
)


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_generate_image_saves_png(monkeypatch, tmp_path: Path):
    payload = {
        "data": [
            {
                "b64_json": base64.b64encode(b"fake-png-bytes").decode("utf-8"),
            }
        ]
    }

    def _fake_urlopen(request, timeout=120):  # noqa: ARG001
        return _FakeResponse(payload)

    monkeypatch.setattr("app.services.image_generation_service.urllib_request.urlopen", _fake_urlopen)

    service = ExperimentalImageGenerationService(
        ImageGenerationConfig(
            enabled=True,
            model="gpt-image-1",
            api_key="test-key",
            base_url="https://example.test/v1",
            output_dir=tmp_path,
        )
    )

    result = service.generate_image(prompt="画一个电磁感应示意图", style="diagram_clean", topic="法拉第")

    assert result["image_url"].startswith("/static/")
    assert result["model"] == "gpt-image-1"
    assert Path(result["image_path"]).exists()
    assert Path(result["image_path"]).read_bytes() == b"fake-png-bytes"


def test_generate_image_requires_configuration(tmp_path: Path):
    service = ExperimentalImageGenerationService(
        ImageGenerationConfig(
            enabled=False,
            model=None,
            api_key=None,
            base_url=None,
            output_dir=tmp_path,
        )
    )

    try:
        service.generate_image(prompt="test", style="diagram_clean", topic=None)
    except ImageGenerationError as exc:
        assert "图片生成功能未配置" in str(exc)
    else:
        raise AssertionError("Expected ImageGenerationError")


def test_get_image_generation_config_falls_back_to_project_settings(monkeypatch, tmp_path: Path):
    class _Section:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Settings:
        llm = _Section(api_key="llm-key", base_url="https://aihubmix.com/v1")
        vision_llm = _Section(api_key=None, base_url=None)

    monkeypatch.delenv("IMAGE_GENERATION_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_IMAGE_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("IMAGE_GENERATION_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_IMAGE_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("IMAGE_GENERATION_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_IMAGE_MODEL", raising=False)
    monkeypatch.setattr("app.services.image_generation_service.load_settings", lambda path: _Settings())  # noqa: ARG005
    monkeypatch.setattr("app.services.image_generation_service.STATIC_DIR", tmp_path)
    _load_project_image_defaults.cache_clear()

    config = get_image_generation_config()

    assert config.enabled is True
    assert config.model == "gpt-image-1-mini"
    assert config.api_key == "llm-key"
    assert config.base_url == "https://aihubmix.com/v1"
