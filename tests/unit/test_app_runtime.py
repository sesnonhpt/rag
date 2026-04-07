from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from starlette.requests import Request

from app.core import app_runtime


@pytest.mark.asyncio
async def test_lifespan_sets_state_and_closes_image_storage(monkeypatch):
    app = FastAPI()
    settings = SimpleNamespace(vector_store=SimpleNamespace(collection_name="default"))
    image_storage_holder = {}

    class _DummyImageStorage:
        def __init__(self, db_path: str, images_root: str):
            self.db_path = db_path
            self.images_root = images_root
            self.closed = False

        def close(self):
            self.closed = True

    class _DummyHistoryStorage:
        def __init__(self, db_path: str):
            self.db_path = db_path

    class _DummyLLMFactory:
        @staticmethod
        def create(_: object):
            return "llm"

    def _build_image_storage(*, db_path: str, images_root: str):
        instance = _DummyImageStorage(db_path=db_path, images_root=images_root)
        image_storage_holder["value"] = instance
        return instance

    monkeypatch.setattr(app_runtime, "load_settings", lambda _: settings)
    monkeypatch.setattr(app_runtime, "build_components", lambda *_: ("hybrid", "reranker"))
    monkeypatch.setattr(app_runtime, "LLMFactory", _DummyLLMFactory)
    monkeypatch.setattr(app_runtime, "ImageStorage", _build_image_storage)
    monkeypatch.setattr(app_runtime, "LessonHistoryStorage", _DummyHistoryStorage)

    async with app_runtime.lifespan(app):
        assert app.state.settings is settings
        assert app.state.hybrid_search == "hybrid"
        assert app.state.reranker == "reranker"
        assert app.state.llm == "llm"
        assert app.state.default_collection == "default"
        assert image_storage_holder["value"].closed is False

    assert image_storage_holder["value"].closed is True


@pytest.mark.asyncio
async def test_global_exception_handler_returns_standard_payload():
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/boom",
        "raw_path": b"/boom",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    request = Request(scope)

    response = await app_runtime.global_exception_handler(request, RuntimeError("boom"))

    assert response.status_code == 500
    assert b"INTERNAL_SERVER_ERROR" in response.body


def test_is_lesson_plan_mock_enabled_defaults_to_false(monkeypatch):
    monkeypatch.delenv("LESSON_PLAN_MOCK_ENABLED", raising=False)

    assert app_runtime.is_lesson_plan_mock_enabled() is False


@pytest.mark.parametrize("raw_value", ["1", "true", "on", "yes"])
def test_is_lesson_plan_mock_enabled_accepts_explicit_opt_in(monkeypatch, raw_value):
    monkeypatch.setenv("LESSON_PLAN_MOCK_ENABLED", raw_value)

    assert app_runtime.is_lesson_plan_mock_enabled() is True
