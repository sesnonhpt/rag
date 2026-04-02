from app.core.lesson_content_helpers import integrate_images_into_markdown
from app.schemas.api_models import LessonImageResource


def _make_image(index: int) -> LessonImageResource:
    return LessonImageResource(
        image_id=f"img-{index}",
        url=f"/lesson-plan-image/img-{index}",
        source="/data/pdf/demo.pdf",
        page=10 + index,
        caption=f"示意图{index}",
    )


def test_integrate_images_prefers_explicit_marker() -> None:
    content = "一、导入\n观察现象（见配图1）\n二、探究"
    output = integrate_images_into_markdown(content, [_make_image(1)])

    assert "见配图1" in output
    assert "![配图1]" in output
    marker_pos = output.find("见配图1")
    image_pos = output.find("![配图1]")
    assert image_pos > marker_pos


def test_integrate_images_anchors_on_generic_figure_cues() -> None:
    content = "1. 如图，分析受力情况。\n2. 解释原因。"
    output = integrate_images_into_markdown(content, [_make_image(1)])

    first_line_pos = output.find("如图")
    image_pos = output.find("![配图1]")
    assert first_line_pos >= 0
    assert image_pos > first_line_pos
