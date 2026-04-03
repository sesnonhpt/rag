from types import SimpleNamespace

from app.core.runtime_helpers import resolve_template_type_from_category
from app.schemas.api_models import LessonPlanRequest
from src.core.templates import TemplateConfig, TemplateManager, TemplateType


def test_lesson_plan_request_exposes_only_current_template_inputs() -> None:
    fields = LessonPlanRequest.model_fields

    assert "template_category" in fields
    assert "notes" in fields
    assert "template_type" not in fields
    assert "grade_level" not in fields
    assert "learning_style" not in fields


def test_resolve_template_type_maps_three_active_categories() -> None:
    assert resolve_template_type_from_category(LessonPlanRequest(topic="示例", template_category="guide").template_category) == "guide_master"
    assert resolve_template_type_from_category(LessonPlanRequest(topic="示例", template_category="teaching_design").template_category) == "teaching_design_master"
    assert resolve_template_type_from_category(LessonPlanRequest(topic="示例", template_category="comprehensive").template_category) == "comprehensive_master"
    assert resolve_template_type_from_category(LessonPlanRequest(topic="示例").template_category) is None


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
