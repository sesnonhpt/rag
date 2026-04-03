from __future__ import annotations

from types import SimpleNamespace

from src.agents.lesson_agent import LessonAgent
from src.agents.models import LessonAgentAssets, LessonAgentState


class _DummyTrace:
    def __init__(self):
        self.stages = []

    def record_stage(self, name, payload, elapsed_ms=None):
        self.stages.append((name, payload, elapsed_ms))


def _build_agent(template_type: str, integrate_images):
    return LessonAgent(
        llm=SimpleNamespace(),
        template_manager=SimpleNamespace(),
        trace=_DummyTrace(),
        request=SimpleNamespace(template_category="guide"),
        resolved_template_type=template_type,
        build_default_prompt=lambda *args, **kwargs: [],
        build_fallback_prompt=lambda *args, **kwargs: [],
        integrate_images=integrate_images,
    )


def test_insert_images_applies_for_guide_template() -> None:
    calls = []

    def _integrate(content, images):
        calls.append((content, images))
        return f"{content}\n[with-images]"

    agent = _build_agent("guide_master", _integrate)
    state = LessonAgentState(
        topic="牛顿第一定律",
        assets=LessonAgentAssets(image_resources=[{"id": "i1"}]),
        draft_content="原文",
    )

    agent._insert_images(state)

    assert state.final_content.endswith("[with-images]")
    assert len(calls) == 1


def test_insert_images_applies_even_when_context_not_usable() -> None:
    calls = []

    def _integrate(content, images):
        calls.append((content, images))
        return "done"

    agent = _build_agent("teaching_design_master", _integrate)
    state = LessonAgentState(
        topic="牛顿第一定律",
        assets=LessonAgentAssets(image_resources=[{"id": "i1"}]),
        draft_content="原文",
    )
    state.metadata["usable_context"] = False

    agent._insert_images(state)

    assert state.final_content == "done"
    assert len(calls) == 1
