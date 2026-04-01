from app.main import app
from fastapi.testclient import TestClient

from src.core.templates.registry import (
    get_default_template_category,
    get_template_categories_payload,
    resolve_template_type_by_category,
)


def test_template_registry_mapping():
    assert resolve_template_type_by_category("comprehensive") == "comprehensive_master"
    assert resolve_template_type_by_category("teaching_design") == "teaching_design_master"
    assert resolve_template_type_by_category("guide") == "guide_master"
    assert get_default_template_category() == "comprehensive"


def test_template_categories_api():
    client = TestClient(app)
    response = client.get("/lesson-template-categories")
    assert response.status_code == 200
    payload = response.json()
    assert "templates" in payload
    assert len(payload["templates"]) == len(get_template_categories_payload())
    assert payload["templates"][0]["category"] == "comprehensive"
