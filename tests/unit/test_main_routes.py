from fastapi.testclient import TestClient

from app.main import app


def test_main_app_basic_routes():
    client = TestClient(app)

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json().get("status") == "ok"

    template_response = client.get("/lesson-template-categories")
    assert template_response.status_code == 200
    payload = template_response.json()
    assert "templates" in payload
    assert len(payload["templates"]) >= 3

    root_response = client.get("/")
    assert root_response.status_code == 200
