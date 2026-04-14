from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.api_models import (
    ImageGenerationExperimentRequest,
    ImageGenerationExperimentResponse,
    ImagePromptSuggestionRequest,
    ImagePromptSuggestionResponse,
)
from app.services.visual_asset_service import suggest_visual_generation_prompt
from app.services.image_generation_service import ExperimentalImageGenerationService, ImageGenerationError

router = APIRouter()


@router.post("/experimental/image-generation", response_model=ImageGenerationExperimentResponse)
async def experimental_image_generation(req: ImageGenerationExperimentRequest):
    service = ExperimentalImageGenerationService()
    try:
        result = service.generate_image(
            prompt=req.prompt,
            style=req.style,
            topic=req.topic,
        )
    except ImageGenerationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ImageGenerationExperimentResponse(
        image_url=result["image_url"],
        image_path=result["image_path"],
        filename=result["filename"],
        model=result["model"],
        style=result["style"],
        prompt=result["prompt"],
        topic=result["topic"],
    )


@router.post("/experimental/image-prompt", response_model=ImagePromptSuggestionResponse)
async def experimental_image_prompt(req: ImagePromptSuggestionRequest):
    from app.main import app

    llm = getattr(app.state, "llm", None)
    prompt, style, source = suggest_visual_generation_prompt(
        topic=req.topic,
        notes=req.notes,
        template_category=req.template_category,
        llm=llm,
    )
    return ImagePromptSuggestionResponse(
        prompt=prompt,
        style=style,
        source=source,
    )
