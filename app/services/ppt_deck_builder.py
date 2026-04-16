"""Builder utilities for editable PPT decks."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.schemas.ppt_models import (
    BuildPptDeckRequest,
    CreatePptDeckRequest,
    LessonDeckModel,
    LessonDeckSourceSnapshotModel,
    LessonDeckThemeModel,
    SlideElementModel,
    LessonSlideModel,
    UpdatePptDeckRequest,
)
from app.services.pptx_export_service import build_default_elements_for_slide, build_lesson_deck


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_deck_from_lesson(req: BuildPptDeckRequest) -> LessonDeckModel:
    built = build_lesson_deck(
        title=req.title,
        content_html=req.content_html,
        image_resources=req.image_resources,
    )
    now = _utc_now()
    slides = [
        LessonSlideModel(
            id=f"slide_{index + 1}_{uuid4().hex[:8]}",
            order=index,
            layout=slide.layout,
            title=slide.title,
            bullets=list(slide.bullets),
            paragraphs=list(slide.paragraphs),
            speaker_notes=list(slide.speaker_notes),
            image_sources=list(slide.image_sources),
            accent_text=slide.accent_text,
            elements=[
                SlideElementModel(**element)
                for element in build_default_elements_for_slide(slide)
            ],
        )
        for index, slide in enumerate(built.slides)
    ]
    return LessonDeckModel(
        deck_id=f"deck_{uuid4().hex}",
        title=built.title,
        topic=req.topic,
        template_category=req.template_category or "ppt",
        slides=slides,
        theme=LessonDeckThemeModel(),
        source_snapshot=LessonDeckSourceSnapshotModel(
            lesson_content=req.lesson_content,
            content_html=req.content_html,
            image_resources=req.image_resources,
        ),
        created_at=now,
        updated_at=now,
    )


def build_deck_from_create_request(req: CreatePptDeckRequest) -> LessonDeckModel:
    now = _utc_now()
    slides = [
        slide.model_copy(update={"order": index})
        for index, slide in enumerate(req.slides)
    ]
    return LessonDeckModel(
        deck_id=f"deck_{uuid4().hex}",
        title=req.title,
        topic=req.topic,
        template_category=req.template_category,
        slides=slides,
        theme=req.theme,
        source_snapshot=req.source_snapshot,
        created_at=now,
        updated_at=now,
    )


def apply_deck_update(deck: LessonDeckModel, req: UpdatePptDeckRequest) -> LessonDeckModel:
    payload = deck.model_dump()
    if req.title is not None:
        payload["title"] = req.title
    if req.topic is not None:
        payload["topic"] = req.topic
    if req.template_category is not None:
        payload["template_category"] = req.template_category
    if req.slides is not None:
        payload["slides"] = [
            slide.model_copy(update={"order": index}).model_dump()
            for index, slide in enumerate(req.slides)
        ]
    if req.theme is not None:
        payload["theme"] = req.theme.model_dump()
    if req.source_snapshot is not None:
        payload["source_snapshot"] = req.source_snapshot.model_dump()
    payload["updated_at"] = req.updated_at or _utc_now()
    return LessonDeckModel.model_validate(payload)
