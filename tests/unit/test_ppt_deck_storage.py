from __future__ import annotations

from pathlib import Path

from app.schemas.ppt_models import LessonDeckModel, LessonSlideModel
from app.services.ppt_deck_storage import PptDeckStorage


def test_ppt_deck_storage_crud(tmp_path: Path):
    storage = PptDeckStorage(db_path=str(tmp_path / "ppt_decks.db"))
    deck = LessonDeckModel(
        deck_id="deck_test",
        title="测试课件",
        slides=[
            LessonSlideModel(
                id="slide_1",
                order=0,
                title="封面",
                layout="cover",
            )
        ],
    )

    storage.save_deck(deck.model_dump(mode="json"))
    loaded = storage.get_deck("deck_test")
    assert loaded is not None
    assert loaded["title"] == "测试课件"

    listing = storage.list_decks(limit=10)
    assert listing
    assert listing[0]["deck_id"] == "deck_test"

    storage.delete_deck("deck_test")
    assert storage.get_deck("deck_test") is None
