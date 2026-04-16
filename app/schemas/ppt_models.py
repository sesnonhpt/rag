"""Pydantic models for editable PPT decks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.api_models import LessonImageResource


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SlideElementModel(BaseModel):
    id: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1, description="text|image|shape")
    x: float = Field(default=0)
    y: float = Field(default=0)
    w: float = Field(default=100)
    h: float = Field(default=40)
    text: Optional[str] = None
    src: Optional[str] = None
    font_size: Optional[int] = None
    bold: bool = False
    fill_color: Optional[str] = None
    text_color: Optional[str] = None
    radius: Optional[int] = None


class LessonSlideModel(BaseModel):
    id: str = Field(..., min_length=1)
    order: int = Field(default=0)
    layout: str = Field(default="standard")
    title: str = Field(..., min_length=1)
    bullets: List[str] = Field(default_factory=list)
    paragraphs: List[str] = Field(default_factory=list)
    speaker_notes: List[str] = Field(default_factory=list)
    image_sources: List[str] = Field(default_factory=list)
    accent_text: Optional[str] = None
    elements: List[SlideElementModel] = Field(default_factory=list)


class LessonDeckThemeModel(BaseModel):
    palette: str = Field(default="blue")
    font_family: str = Field(default="Microsoft YaHei")
    aspect_ratio: str = Field(default="16:9")


class LessonDeckSourceSnapshotModel(BaseModel):
    lesson_content: Optional[str] = None
    content_html: Optional[str] = None
    image_resources: List[LessonImageResource] = Field(default_factory=list)


class LessonDeckModel(BaseModel):
    deck_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    topic: Optional[str] = None
    template_category: str = Field(default="ppt")
    slides: List[LessonSlideModel] = Field(default_factory=list)
    theme: LessonDeckThemeModel = Field(default_factory=LessonDeckThemeModel)
    source_snapshot: LessonDeckSourceSnapshotModel = Field(default_factory=LessonDeckSourceSnapshotModel)
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)


class BuildPptDeckRequest(BaseModel):
    title: str = Field(..., min_length=1)
    topic: Optional[str] = None
    content_html: str = Field(..., min_length=1)
    lesson_content: Optional[str] = None
    image_resources: List[LessonImageResource] = Field(default_factory=list)
    template_category: str = Field(default="ppt")


class CreatePptDeckRequest(BaseModel):
    title: str = Field(..., min_length=1)
    topic: Optional[str] = None
    template_category: str = Field(default="ppt")
    slides: List[LessonSlideModel] = Field(default_factory=list)
    theme: LessonDeckThemeModel = Field(default_factory=LessonDeckThemeModel)
    source_snapshot: LessonDeckSourceSnapshotModel = Field(default_factory=LessonDeckSourceSnapshotModel)


class UpdatePptDeckRequest(BaseModel):
    title: Optional[str] = None
    topic: Optional[str] = None
    template_category: Optional[str] = None
    slides: Optional[List[LessonSlideModel]] = None
    theme: Optional[LessonDeckThemeModel] = None
    source_snapshot: Optional[LessonDeckSourceSnapshotModel] = None
    updated_at: Optional[str] = None


class PptDeckListItem(BaseModel):
    deck_id: str
    title: str
    topic: Optional[str] = None
    template_category: str = "ppt"
    slide_count: int = 0
    created_at: str
    updated_at: str


class PptDeckListResponse(BaseModel):
    decks: List[PptDeckListItem] = Field(default_factory=list)


class PptDeckResponse(BaseModel):
    deck: LessonDeckModel


class PptDeckDeleteResponse(BaseModel):
    ok: bool = True
    deck_id: str
