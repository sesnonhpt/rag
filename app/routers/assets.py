from __future__ import annotations

from io import BytesIO
import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from app.core.lesson_content_helpers import find_image_file_by_id, resolve_image_file_path
from app.core.paths import STATIC_DIR
from app.schemas.api_models import HealthResponse
from src.observability.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

_WEB_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
_CONVERTIBLE_EXTENSIONS = {".jpx", ".jp2", ".j2k", ".jpf", ".tif", ".tiff"}


def _stream_web_compatible_image(path: Path):
    suffix = path.suffix.lower()
    if suffix in _WEB_IMAGE_EXTENSIONS:
        media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type)

    if suffix in _CONVERTIBLE_EXTENSIONS:
        try:
            from PIL import Image, ImageOps

            with Image.open(path) as image:
                normalized = ImageOps.exif_transpose(image)
                if normalized.mode not in {"RGB", "L"}:
                    normalized = normalized.convert("RGB")
                buffer = BytesIO()
                normalized.save(buffer, format="PNG")
                buffer.seek(0)
                return StreamingResponse(buffer, media_type="image/png")
        except Exception as exc:
            logger.warning("lesson_image.convert_failed path=%s error=%r", path, exc)

    return FileResponse(path, media_type="application/octet-stream")


@router.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_file = STATIC_DIR / "lesson-plan.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return HTMLResponse(
        content=html_file.read_text(encoding="utf-8"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/lesson-plan.html", response_class=HTMLResponse)
async def serve_lesson_plan_ui():
    html_file = STATIC_DIR / "lesson-plan.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="Lesson Plan UI file not found")
    return HTMLResponse(
        content=html_file.read_text(encoding="utf-8"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/lesson-plan-image/{image_id}")
async def serve_lesson_plan_image(image_id: str, request: Request):
    image_storage = request.app.state.image_storage
    image_path = image_storage.get_image_path(image_id)
    path = resolve_image_file_path(image_path) if image_path else Path()

    if (not image_path) or (not path.exists()) or (not path.is_file()):
        fallback = find_image_file_by_id(image_id)
        if fallback is None:
            raise HTTPException(status_code=404, detail="Image not found")
        path = fallback
        try:
            image_storage.register_image(
                image_id=image_id,
                file_path=path,
                collection=path.parent.parent.name if path.parent.parent != path.parent else "default",
                doc_hash=path.parent.name if path.parent != path else None,
                page_num=None,
            )
        except Exception:
            logger.warning("lesson_image.reregister_failed image_id=%s path=%s", image_id, path)

    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Image file not found")

    return _stream_web_compatible_image(path)


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    state = request.app.state
    return HealthResponse(
        status="ok",
        components={
            "hybrid_search": getattr(state, "hybrid_search", None) is not None,
            "reranker": getattr(state, "reranker", None) is not None,
            "llm": getattr(state, "llm", None) is not None,
        },
    )
