from __future__ import annotations

from app.schemas.ppt_models import BuildPptDeckRequest, UpdatePptDeckRequest
from app.services.ppt_deck_builder import apply_deck_update, build_deck_from_lesson


def test_build_deck_from_lesson_creates_structured_slides():
    deck = build_deck_from_lesson(
        BuildPptDeckRequest(
            title="测试课件",
            topic="牛顿第三定律",
            content_html=(
                "<h1>《牛顿第三定律》PPT课件</h1>"
                "<h2>幻灯片1｜学习目标</h2>"
                "<ul><li>理解作用力与反作用力</li></ul>"
                "<h2>幻灯片2｜课堂小结</h2>"
                "<p>讲解提示：最后回到异体这一判断点。</p>"
            ),
            image_resources=[],
        )
    )

    assert deck.deck_id.startswith("deck_")
    assert deck.title == "《牛顿第三定律》PPT课件"
    assert len(deck.slides) >= 2
    assert deck.slides[0].id
    assert deck.slides[0].order == 0
    assert deck.slides[0].elements
    assert deck.slides[0].elements[0].type in {"shape", "text"}


def test_apply_deck_update_reorders_slides_and_updates_title():
    deck = build_deck_from_lesson(
        BuildPptDeckRequest(
            title="测试课件",
            topic="牛顿第三定律",
            content_html=(
                "<h1>《牛顿第三定律》PPT课件</h1>"
                "<h2>幻灯片1｜学习目标</h2><ul><li>A</li></ul>"
                "<h2>幻灯片2｜课堂小结</h2><ul><li>B</li></ul>"
            ),
        )
    )

    reordered = [deck.slides[1], deck.slides[0]]
    updated = apply_deck_update(
        deck,
        UpdatePptDeckRequest(
            title="新版课件",
            slides=reordered,
        ),
    )

    assert updated.title == "新版课件"
    assert updated.slides[0].order == 0
    assert updated.slides[1].order == 1
