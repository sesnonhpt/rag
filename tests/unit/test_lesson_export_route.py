from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.routers.lesson import export_lesson_plan_docx
from app.schemas.api_models import ExportDocxRequest


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
