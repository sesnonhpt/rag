from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.routers.lesson import export_lesson_plan_docx, export_lesson_plan_pptx
from app.routers.ppt import create_ppt_deck_from_lesson
from app.schemas.api_models import ExportDocxRequest, ExportPptxRequest
from app.schemas.ppt_models import BuildPptDeckRequest
from app.services.lesson_service import _normalize_ppt_lesson_content
from app.services.ppt_deck_storage import PptDeckStorage


def test_export_lesson_plan_docx_returns_docx_response(monkeypatch):
    monkeypatch.setattr(
        "app.routers.lesson.build_lesson_docx_bytes",
        lambda **kwargs: b"DOCX_BYTES",
    )

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(image_storage=None)))
    response = asyncio.run(
        export_lesson_plan_docx(
            ExportDocxRequest(title="测试 教案", content_html="<h1>标题</h1><p>内容</p>"),
            request,
        )
    )

    assert response.status_code == 200
    assert response.body == b"DOCX_BYTES"
    assert (
        response.headers["content-type"]
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert response.headers["content-length"] == str(len(b"DOCX_BYTES"))
    assert "attachment; filename=" in response.headers["content-disposition"]
    assert "filename*=UTF-8''" in response.headers["content-disposition"]


def test_export_lesson_plan_pptx_returns_pptx_response(monkeypatch):
    monkeypatch.setattr(
        "app.routers.lesson.build_lesson_pptx_bytes",
        lambda **kwargs: b"PPTX_BYTES",
    )

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(image_storage=None)))
    response = asyncio.run(
        export_lesson_plan_pptx(
            ExportPptxRequest(title="测试 课件", content_html="<h1>标题</h1><p>内容</p>"),
            request,
        )
    )

    assert response.status_code == 200
    assert response.body == b"PPTX_BYTES"
    assert (
        response.headers["content-type"]
        == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )
    assert response.headers["content-length"] == str(len(b"PPTX_BYTES"))
    assert "attachment; filename=" in response.headers["content-disposition"]
    assert "filename*=UTF-8''" in response.headers["content-disposition"]


def test_create_ppt_deck_from_lesson_persists_and_returns_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.routers.ppt.build_deck_from_lesson",
        lambda req: SimpleNamespace(
            deck_id="deck_test",
            title=req.title,
            topic=req.topic,
            template_category=req.template_category,
            slides=[{"id": "slide_1"}, {"id": "slide_2"}],
            model_dump=lambda mode="json": {
                "deck_id": "deck_test",
                "title": req.title,
                "topic": req.topic,
                "template_category": req.template_category,
                "slides": [{"id": "slide_1"}, {"id": "slide_2"}],
            },
        ),
    )

    storage = PptDeckStorage(db_path=str(tmp_path / "ppt_decks.db"))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(ppt_deck_storage=storage)))

    response = asyncio.run(
        create_ppt_deck_from_lesson(
            BuildPptDeckRequest(
                title="测试课件",
                topic="牛顿第三定律",
                content_html="<h1>测试</h1>",
                template_category="ppt",
            ),
            request,
        )
    )

    assert response.deck_id == "deck_test"
    assert response.title == "测试课件"
    assert response.slide_count == 2

    saved = storage.get_deck("deck_test")
    assert saved is not None
    assert saved["title"] == "测试课件"


def test_normalize_ppt_lesson_content_rewrites_comprehensive_title():
    content = "# 《牛顿第三定律》综合模版\n## 幻灯片1｜学习目标\n- 理解概念\n"

    normalized = _normalize_ppt_lesson_content(content, "牛顿第三定律")

    assert normalized.startswith("# 《牛顿第三定律》PPT课件")
    assert "综合模版" not in normalized
