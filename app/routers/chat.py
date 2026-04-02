from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas.api_models import ChatRequest, ChatResponse
from app.services.chat_service import generate_chat_response

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    return await generate_chat_response(req, request)
