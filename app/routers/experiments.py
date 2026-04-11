from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.api_models import ImageGenerationExperimentRequest, ImageGenerationExperimentResponse
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
