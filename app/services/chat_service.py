"""Chat service layer for retrieval + generation flow."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import HTTPException
from app.core.lesson_content_helpers import sanitize_source_path
from app.core.prompt_builders import build_chat_prompt
from app.core.runtime_helpers import build_api_error_detail
from app.schemas.api_models import ChatResponse, Citation
from src.core.trace import TraceContext
from src.libs.llm.base_llm import Message
from src.observability.logger import get_logger

logger = get_logger(__name__)


async def generate_chat_response(req: Any, request: Any) -> Any:
    state = request.app.state
    settings = state.settings
    hybrid_search = state.hybrid_search
    reranker = state.reranker
    llm = state.llm

    top_k = req.top_k or settings.retrieval.fusion_top_k
    trace = TraceContext(trace_type="chat")

    try:
        # Offload retrieval to a worker thread because the underlying
        # retrievers may perform blocking CPU/I/O work.
        hybrid_result = await asyncio.to_thread(
            hybrid_search.search,
            query=req.question,
            top_k=top_k,
            filters=None,
            trace=trace,
        )
    except Exception as e:
        logger.exception("HybridSearch failed")
        raise HTTPException(
            status_code=500,
            detail=build_api_error_detail(
                code="RETRIEVAL_ERROR",
                message="检索服务异常，请稍后重试",
                stage="chat_retrieval",
                trace_id=trace.trace_id,
            ),
        ) from e

    results = hybrid_result if not hasattr(hybrid_result, "results") else hybrid_result.results

    if results and req.use_rerank and reranker.is_enabled:
        try:
            rerank_result = await asyncio.to_thread(
                reranker.rerank,
                query=req.question,
                results=results,
                top_k=top_k,
                trace=trace,
            )
            results = rerank_result.results
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")

    citations = [
        Citation(
            source=sanitize_source_path((r.metadata or {}).get("source_path", "unknown")),
            score=round(float((r.metadata or {}).get("original_score", r.score) or 0.0), 4),
            text=(r.text or "")[:200],
        )
        for r in results
    ]

    try:
        if results:
            messages = build_chat_prompt(req.question, results)
        else:
            messages = [
                Message(
                    role="system",
                    content=(
                        "你是一个知识库问答助手。当前知识库中没有找到与用户问题相关的内容，"
                        "请基于你自己的知识回答用户问题。请明确告知用户你是基于自身知识回答的，"
                        "而不是基于知识库内容。"
                    ),
                ),
                Message(
                    role="user",
                    content=f"问题：{req.question}",
                ),
            ]
        # Provider SDK calls are synchronous, so run generation outside
        # the event loop to keep concurrent requests moving.
        llm_response = await asyncio.to_thread(llm.chat, messages)
        answer = llm_response.content
    except Exception as e:
        logger.exception("LLM generation failed")
        raise HTTPException(
            status_code=500,
            detail=build_api_error_detail(
                code="LLM_SERVICE_ERROR",
                message=f"LLM 服务异常: {str(e)}"[:280],
                stage="chat_generation",
                trace_id=trace.trace_id,
            ),
        ) from e

    return ChatResponse(answer=answer, citations=citations)
