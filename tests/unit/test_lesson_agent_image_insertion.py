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


def test_teacher_note_is_appended_to_generation_messages() -> None:
    agent = LessonAgent(
        llm=SimpleNamespace(),
        template_manager=SimpleNamespace(build_prompt=lambda **kwargs: [SimpleNamespace(role="system", content="s"), SimpleNamespace(role="user", content="u")]),
        trace=_DummyTrace(),
        request=SimpleNamespace(
            template_category="guide",
            include_background=True,
            include_facts=True,
            include_examples=True,
            topic="牛顿第一定律",
            notes="希望加入一个贴近生活的汽车案例，并减少纯理论表述。",
        ),
        resolved_template_type="guide_master",
        build_default_prompt=lambda *args, **kwargs: [],
        build_fallback_prompt=lambda *args, **kwargs: [],
        integrate_images=lambda content, images: content,
    )
    state = LessonAgentState(
        topic="牛顿第一定律",
        assets=LessonAgentAssets(text_results=[SimpleNamespace(text="资料")]),
    )

    messages = agent._build_generation_messages(state)

    assert messages[-1].role == "user"
    assert "老师备注" in messages[-1].content
    assert "汽车案例" in messages[-1].content


def test_teacher_note_hard_constraints_include_compare_and_newton_story() -> None:
    agent = LessonAgent(
        llm=SimpleNamespace(),
        template_manager=SimpleNamespace(),
        trace=_DummyTrace(),
        request=SimpleNamespace(
            template_category="comprehensive",
            topic="牛顿第二定律",
            notes="希望与牛顿第一定律比较，同时引入牛顿的故事。",
        ),
        resolved_template_type="comprehensive_master",
        build_default_prompt=lambda *args, **kwargs: [],
        build_fallback_prompt=lambda *args, **kwargs: [],
        integrate_images=lambda content, images: content,
    )

    guidance = agent._teacher_note_hard_constraints(agent.request.notes)

    assert "与牛顿第一定律的联系与区别" in guidance
    assert "牛顿生平或科学史故事" in guidance


def test_teacher_note_requirement_issues_detects_missing_compare_and_story() -> None:
    agent = LessonAgent(
        llm=SimpleNamespace(),
        template_manager=SimpleNamespace(),
        trace=_DummyTrace(),
        request=SimpleNamespace(
            template_category="comprehensive",
            topic="牛顿第二定律",
            notes="希望与牛顿第一定律比较，同时引入牛顿的故事。",
        ),
        resolved_template_type="comprehensive_master",
        build_default_prompt=lambda *args, **kwargs: [],
        build_fallback_prompt=lambda *args, **kwargs: [],
        integrate_images=lambda content, images: content,
    )

    issues = agent._teacher_note_requirement_issues(
        "## 一、教学目标\n本课介绍牛顿第二定律。\n## 情境导入\n简单导入。"
    )

    assert "缺少“与牛顿第一定律的联系与区别”专门小节" in issues
    assert "导入环节缺少服务于教学导入的牛顿故事或生平材料" in issues
