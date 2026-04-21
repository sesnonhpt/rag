from __future__ import annotations

import re
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from app.core.lesson_content_helpers import resolve_docx_image_path
from app.core.runtime_helpers import build_api_error_detail
from app.schemas.ppt_models import (
    BuildPptDeckRequest,
    CreatePptDeckRequest,
    PptDeckCreateFromLessonResponse,
    PptDeckDeleteResponse,
    PptDeckListResponse,
    PptDeckListItem,
    PptDeckResponse,
    UpdatePptDeckRequest,
)
from app.services.ppt_deck_builder import (
    apply_deck_update,
    build_deck_from_create_request,
    build_deck_from_lesson,
)
from app.services.pptx_export_service import build_lesson_pptx_bytes_from_deck
from src.observability.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _ensure_ppt_template_category(template_category: str | None) -> None:
    if str(template_category or "").strip() == "ppt":
        return
    raise HTTPException(status_code=400, detail="只有 PPT 模版才能进入 PPT 预览")


@router.post("/lesson-plan/build-ppt-deck", response_model=PptDeckResponse)
async def build_ppt_deck(req: BuildPptDeckRequest):
    _ensure_ppt_template_category(req.template_category)
    deck = build_deck_from_lesson(req)
    return PptDeckResponse(deck=deck)


@router.post("/lesson-plan/create-ppt-deck", response_model=PptDeckCreateFromLessonResponse)
async def create_ppt_deck_from_lesson(req: BuildPptDeckRequest, request: Request):
    _ensure_ppt_template_category(req.template_category)
    storage = request.app.state.ppt_deck_storage
    deck = build_deck_from_lesson(req)
    storage.save_deck(deck.model_dump(mode="json"))
    return PptDeckCreateFromLessonResponse(
        deck_id=deck.deck_id,
        title=deck.title,
        topic=deck.topic,
        template_category=deck.template_category,
        slide_count=len(deck.slides),
    )


@router.get("/ppt-decks", response_model=PptDeckListResponse)
async def list_ppt_decks(request: Request, limit: int = 20):
    storage = request.app.state.ppt_deck_storage
    records = storage.list_decks(limit=limit)
    return PptDeckListResponse(decks=[PptDeckListItem(**item) for item in records])


@router.post("/ppt-decks", response_model=PptDeckResponse)
async def create_ppt_deck(req: CreatePptDeckRequest, request: Request):
    storage = request.app.state.ppt_deck_storage
    deck = build_deck_from_create_request(req)
    storage.save_deck(deck.model_dump(mode="json"))
    return PptDeckResponse(deck=deck)


@router.get("/ppt-decks/{deck_id}", response_model=PptDeckResponse)
async def get_ppt_deck(deck_id: str, request: Request):
    storage = request.app.state.ppt_deck_storage
    payload = storage.get_deck(deck_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="PPT deck not found")
    _ensure_ppt_template_category(payload.get("template_category"))
    return PptDeckResponse(deck=payload)


@router.put("/ppt-decks/{deck_id}", response_model=PptDeckResponse)
async def update_ppt_deck(deck_id: str, req: UpdatePptDeckRequest, request: Request):
    storage = request.app.state.ppt_deck_storage
    payload = storage.get_deck(deck_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="PPT deck not found")
    current = PptDeckResponse(deck=payload).deck
    updated = apply_deck_update(current, req)
    storage.save_deck(updated.model_dump(mode="json"))
    return PptDeckResponse(deck=updated)


@router.delete("/ppt-decks/{deck_id}", response_model=PptDeckDeleteResponse)
async def delete_ppt_deck(deck_id: str, request: Request):
    storage = request.app.state.ppt_deck_storage
    storage.delete_deck(deck_id)
    return PptDeckDeleteResponse(deck_id=deck_id)


@router.post("/ppt-decks/{deck_id}/export-pptx")
async def export_ppt_deck(deck_id: str, request: Request):
    storage = request.app.state.ppt_deck_storage
    payload = storage.get_deck(deck_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="PPT deck not found")
    _ensure_ppt_template_category(payload.get("template_category"))

    deck = PptDeckResponse(deck=payload).deck
    try:
        image_storage = getattr(request.app.state, "image_storage", None)
        filename = re.sub(r'[\\/:*?"<>|]+', "_", deck.title).strip() or "教案课件"
        pptx_bytes = build_lesson_pptx_bytes_from_deck(
            deck=deck,
            resolve_image_path=lambda src: resolve_docx_image_path(src, image_storage),
        )
        ascii_filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._") or "lesson-slides"
        headers = {
            "Content-Disposition": (
                f'attachment; filename="{ascii_filename}.pptx"; '
                f"filename*=UTF-8''{quote(filename, safe='')}.pptx"
            ),
            "Content-Length": str(len(pptx_bytes)),
        }
        return Response(
            content=pptx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers=headers,
        )
    except Exception as e:
        logger.exception("PPT deck export failed deck_id=%s", deck_id)
        error_text = str(e).strip() or repr(e)
        raise HTTPException(
            status_code=500,
            detail=build_api_error_detail(
                code="PPT_DECK_EXPORT_ERROR",
                message=f"PPT deck 导出失败: {error_text}"[:280],
                stage="ppt_deck_export",
            ),
        ) from e
