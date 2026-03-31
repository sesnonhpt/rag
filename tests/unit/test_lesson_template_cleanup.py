from types import SimpleNamespace

from app.chat_api import LessonPlanRequest, _resolve_template_type
from src.core.templates import TemplateConfig, TemplateManager, TemplateType


def test_lesson_plan_request_exposes_only_current_template_inputs() -> None:
    fields = LessonPlanRequest.model_fields

    assert "template_category" in fields
    assert "template_type" not in fields
    assert "grade_level" not in fields
    assert "learning_style" not in fields


def test_resolve_template_type_maps_three_active_categories() -> None:
    assert _resolve_template_type(LessonPlanRequest(topic="示例", template_category="guide")) == "guide_master"
    assert _resolve_template_type(LessonPlanRequest(topic="示例", template_category="teaching_design")) == "teaching_design_master"
    assert _resolve_template_type(LessonPlanRequest(topic="示例", template_category="comprehensive")) == "comprehensive_master"
    assert _resolve_template_type(LessonPlanRequest(topic="示例")) is None


def test_template_manager_builds_prompts_for_all_active_templates() -> None:
    manager = TemplateManager()
    contexts = [SimpleNamespace(text="课堂示例文本")]

    for template_type in (
        TemplateType.GUIDE_MASTER,
        TemplateType.TEACHING_DESIGN_MASTER,
        TemplateType.COMPREHENSIVE_MASTER,
    ):
        messages = manager.build_prompt(
            config=TemplateConfig(template_type=template_type),
            topic="动物儿歌",
            contexts=contexts,
            retrieved_images=[],
        )

        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
