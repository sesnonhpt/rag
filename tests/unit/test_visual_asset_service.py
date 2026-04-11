from __future__ import annotations

from app.schemas.api_models import LessonImageResource
from app.services.visual_asset_service import (
    build_visual_generation_prompt,
    detect_visual_generation_intent,
    maybe_generate_visual_asset,
    should_generate_visual_asset,
    should_auto_generate_visuals_from_notes,
)


def test_should_auto_generate_visuals_from_notes_detects_visual_intent() -> None:
    assert should_auto_generate_visuals_from_notes("希望多一点图片，并补一张示意图") is True


def test_should_auto_generate_visuals_from_notes_respects_negative_intent() -> None:
    assert should_auto_generate_visuals_from_notes("纯文字即可，不需要AI示意图") is False


def test_detect_visual_generation_intent_uses_llm_classification() -> None:
    class _FakeLLM:
        def chat(self, messages):
            assert "备注：想让课堂更直观一些，最好有图辅助理解" in messages[-1].content
            return type("Resp", (), {"content": '{"generate_visual": true, "reason": "用户希望图示辅助理解"}'})()

    assert detect_visual_generation_intent("想让课堂更直观一些，最好有图辅助理解", llm=_FakeLLM()) is True


def test_detect_visual_generation_intent_falls_back_to_heuristic_on_llm_error() -> None:
    class _BrokenLLM:
        def chat(self, messages):  # noqa: ARG002
            raise RuntimeError("llm unavailable")

    assert detect_visual_generation_intent("希望多一点图片，并补一张示意图", llm=_BrokenLLM()) is True


def test_should_generate_visual_asset_requires_visual_notes() -> None:
    assert should_generate_visual_asset(
        topic="法拉第与电磁感应",
        notes="请重点讲清楚原理，不需要图片",
        template_category="comprehensive",
        existing_images=[],
    ) is False


def test_should_generate_visual_asset_limits_supported_templates() -> None:
    assert should_generate_visual_asset(
        topic="法拉第与电磁感应",
        notes="增加一张示意图",
        template_category="guide",
        existing_images=[],
    ) is False


def test_should_generate_visual_asset_ignores_existing_images_when_notes_request_visuals() -> None:
    assert should_generate_visual_asset(
        topic="法拉第与电磁感应",
        notes="希望多一点图片，图文并茂一些",
        template_category="comprehensive",
        llm=None,
        existing_images=[
            LessonImageResource(
                image_id="img-1",
                url="/lesson-plan-image/img-1",
                source="/data/pdf/demo.pdf",
            ),
            LessonImageResource(
                image_id="img-2",
                url="/lesson-plan-image/img-2",
                source="/data/pdf/demo.pdf",
            ),
        ],
    ) is True


def test_build_visual_generation_prompt_prefers_illustration_for_classroom_notes() -> None:
    prompt, style = build_visual_generation_prompt(
        topic="法拉第与电磁感应",
        notes="做课堂导入，增加情境插画感",
        template_category="comprehensive",
    )

    assert "法拉第与电磁感应" in prompt
    assert style == "education_illustration"


def test_maybe_generate_visual_asset_returns_generated_resource(monkeypatch) -> None:
    class _FakeService:
        def generate_image(self, *, prompt: str, style: str, topic: str | None):
            assert "教学示意图" in prompt
            return {
                "image_url": "/static/generated-images/test.png",
                "image_path": "/tmp/test.png",
                "filename": "exp-test.png",
                "model": "gpt-image-1-mini",
                "style": style,
                "prompt": prompt,
                "topic": topic or "",
            }

    monkeypatch.setattr("app.services.visual_asset_service.ExperimentalImageGenerationService", _FakeService)

    resource = maybe_generate_visual_asset(
        topic="法拉第与电磁感应",
        notes="增加一张示意图",
        template_category="comprehensive",
        existing_images=[],
        llm=None,
    )

    assert resource is not None
    assert isinstance(resource, LessonImageResource)
    assert resource.source_type == "generated"
    assert resource.role == "supporting_visual"
    assert resource.model == "gpt-image-1-mini"
