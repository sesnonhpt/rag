from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_client_config_exposes_image_generation_flag(monkeypatch):
    monkeypatch.setattr("app.routers.assets.get_image_generation_config", lambda: type("Cfg", (), {"enabled": True})())

    client = TestClient(app)
    response = client.get("/client-config")

    assert response.status_code == 200
    assert response.json()["image_generation_enabled"] is True


def test_experimental_image_generation_route(monkeypatch):
    class _FakeService:
        def generate_image(self, *, prompt: str, style: str, topic: str | None):
            assert "法拉第" in prompt
            return {
                "image_url": "/static/generated-images/test.png",
                "image_path": "/tmp/test.png",
                "filename": "test.png",
                "model": "gpt-image-1",
                "style": style,
                "prompt": prompt,
                "topic": topic or "",
            }

    monkeypatch.setattr("app.routers.experiments.ExperimentalImageGenerationService", _FakeService)

    client = TestClient(app)
    response = client.post(
        "/experimental/image-generation",
        json={
            "prompt": "请生成法拉第电磁感应概念示意图",
            "topic": "法拉第与电磁感应",
            "style": "diagram_clean",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["image_url"] == "/static/generated-images/test.png"
    assert payload["style"] == "diagram_clean"
