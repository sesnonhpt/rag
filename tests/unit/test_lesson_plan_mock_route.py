from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.routers.lesson import generate_mock_lesson_plan
from app.schemas.api_models import LessonPlanRequest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("notes", "template_category"),
    [
        (None, "guide"),
        (None, "teaching_design"),
        (None, "comprehensive"),
        ("增加生活案例、增加图片", "guide"),
        ("增加生活案例、增加图片", "teaching_design"),
        ("增加生活案例、增加图片", "comprehensive"),
    ],
)
async def test_generate_mock_lesson_plan_supports_newton_third_law_matrix(monkeypatch, notes, template_category):
    async def _skip_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("app.routers.lesson.asyncio.sleep", _skip_sleep)
    monkeypatch.setenv("LESSON_PLAN_MOCK_ENABLED", "1")

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    response = await generate_mock_lesson_plan(
        LessonPlanRequest(
            topic="牛顿第三定律",
            notes=notes,
            template_category=template_category,
        ),
        request,
    )

    assert response.topic == "牛顿第三定律"
    assert response.execution_plan["mode"] == "mock"
    assert response.execution_plan["template_category"] == template_category
    assert response.lesson_content
