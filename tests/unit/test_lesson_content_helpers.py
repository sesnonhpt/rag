from pathlib import Path

from app.core import lesson_content_helpers as helpers
from app.core.lesson_content_helpers import integrate_images_into_markdown, renumber_image_references
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
    assert "**配图1：示意图1**" in output
    assert "来源：/data/pdf/demo.pdf · 第 11 页" in output
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


def test_integrate_generated_images_marks_ai_source() -> None:
    content = "一、导入\n二、探究"
    generated = LessonImageResource(
        image_id="generated-1",
        url="/static/generated-images/demo.png",
        source="AI 示意图",
        caption="AI 生成教学示意图：法拉第与电磁感应",
        source_type="generated",
        role="supporting_visual",
        model="gpt-image-1-mini",
    )

    output = integrate_images_into_markdown(content, [generated])

    assert "AI 生成教学示意图" in output
    assert "AI 示意图 · gpt-image-1-mini" in output


def test_integrate_images_same_line_preserves_marker_order() -> None:
    content = "讲解作用力与反作用力时，可结合配图3、配图4进行辨析。"
    output = integrate_images_into_markdown(content, [_make_image(3), _make_image(4)], start_index=3)

    image3_pos = output.find("![配图3]")
    image4_pos = output.find("![配图4]")
    assert image3_pos >= 0
    assert image4_pos > image3_pos


def test_integrate_images_does_not_insert_remaining_blocks_inside_existing_image_block() -> None:
    content = "结合配图1讲解作用力与反作用力。\n\n继续说明：如图，比较更多现象。"
    output = integrate_images_into_markdown(content, [_make_image(1), _make_image(2)], start_index=1)

    block1_title = output.find("**配图1：示意图1**")
    block1_image = output.find("![配图1]")
    block2_title = output.find("**配图2：示意图2**")

    assert block1_title >= 0
    assert block1_image > block1_title
    assert block2_title > block1_image


def test_integrate_images_append_strategy_keeps_generated_visual_after_referenced_blocks() -> None:
    content = "实验一：两人互拉弹簧测力计（配图1）。"
    output = integrate_images_into_markdown(content, [_make_image(7)], start_index=7, placement_strategy="append")

    assert output.startswith("实验一：两人互拉弹簧测力计。")
    assert output.find("**配图7：示意图7**") > output.find("实验一：两人互拉弹簧测力计。")


def test_renumber_image_references_reorders_to_start_from_one() -> None:
    content = "**配图4：教材配图**\n\n![配图4](/a.png)\n\n参考配图4，并对比配图7。\n\n**配图7：AI 示意图**"

    output = renumber_image_references(content)

    assert "配图1：教材配图" in output
    assert "![配图1]" in output
    assert "参考配图1，并对比配图2。" in output
    assert "配图2：AI 示意图" in output


def test_renumber_image_references_prefers_rendered_block_order_over_earlier_inline_mentions() -> None:
    content = (
        "结合配图1讲解概念。\n\n"
        "**配图2：教材配图**\n\n"
        "![配图2](/b.png)\n\n"
        "**配图1：教材配图**\n\n"
        "![配图1](/a.png)\n"
    )

    output = renumber_image_references(content)

    assert output.find("**配图1：教材配图**") < output.find("**配图2：教材配图**")
    assert "结合配图2讲解概念。" in output


def test_low_quality_image_file_detection(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "tiny.png"

    class _FakeImage:
        size = (80, 80)

        def convert(self, mode):  # noqa: ARG002
            return self

        def thumbnail(self, size):  # noqa: ARG002
            return None

        def getdata(self):
            return [10] * (80 * 80)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("PIL.Image.open", lambda path: _FakeImage())

    assert helpers._is_low_quality_lesson_image_file(image_path) is True
