"""Lesson orchestration service layer."""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

from app.core.lesson_content_helpers import (
    extract_image_resources,
    integrate_images_into_markdown,
    is_result_relevant_to_topic,
    looks_like_lesson_refusal,
    prioritize_visual_results,
    sanitize_source_path,
    to_review_report_response,
    renumber_image_references,
)
from app.core.prompt_builders import build_lesson_plan_prompt, build_lesson_plan_prompt_fallback
from app.core.runtime_helpers import (
    is_fast_mode_enabled,
    is_planner_llm_enabled,
    resolve_llm_auth_for_model,
    resolve_template_type_from_category,
)
from app.schemas.api_models import Citation, LessonPlanResponse
from app.services.visual_asset_service import maybe_generate_visual_asset
from src.agents import (
    ConversationAgent,
    LessonOrchestrator,
    PlannerAgent,
    QueryAgent,
    RetrieverAgent,
    WriterReviewerAgent,
)
from src.core.trace import TraceContext
from src.core.templates import TemplateManager, get_template_label_by_category
from src.libs.llm.base_llm import Message
from src.observability.logger import get_logger

logger = get_logger(__name__)


def generate_lesson_plan_internal(
    req: Any,
    request: Any,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Any:
    started = time.monotonic()
    state = request.app.state
    settings = state.settings
    hybrid_search = state.hybrid_search
    reranker = state.reranker
    fast_mode = is_fast_mode_enabled()
    logger.info(
        "lesson_plan.internal_start topic=%s template_category=%s model=%s fast_mode=%s collection=%s",
        req.topic,
        req.template_category,
        req.model or state.settings.llm.model,
        fast_mode,
        req.collection,
    )
    if progress_callback is not None:
        progress_callback(
            "internal_start",
            {
                "topic": req.topic,
                "template_category": req.template_category or "comprehensive",
                "model": req.model or state.settings.llm.model,
                "collection": req.collection,
            },
        )

    if req.model:
        from dataclasses import replace as dc_replace
        from src.core.settings import LLMSettings
        from src.libs.llm.llm_factory import LLMFactory

        resolved_api_key, resolved_base_url = resolve_llm_auth_for_model(req.model, settings)
        new_llm_config = LLMSettings(
            provider=settings.llm.provider,
            model=req.model,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )
        new_settings = dc_replace(settings, llm=new_llm_config)
        llm = LLMFactory.create(new_settings)
        logger.info(f"Using custom model: {req.model}")
    else:
        llm = state.llm

    top_k = int(os.environ.get("LESSON_FAST_TOP_K", "8")) if fast_mode else 15
    trace = TraceContext(trace_type="lesson_plan")
    resolved_template_type = resolve_template_type_from_category(req.template_category)
    conversation_agent = ConversationAgent()
    planner_agent = PlannerAgent(llm=llm if (not fast_mode and is_planner_llm_enabled()) else None)
    query_agent = QueryAgent()
    conversation_state = conversation_agent.prepare_state(req, req.conversation_state)
    retriever_agent = RetrieverAgent(
        hybrid_search=hybrid_search,
        reranker=reranker,
        trace=trace,
        top_k=top_k,
        prioritize_visual_results=prioritize_visual_results,
        relevance_check=is_result_relevant_to_topic,
        extract_image_resources=extract_image_resources,
        sanitize_source_path=sanitize_source_path,
        image_storage=request.app.state.image_storage,
        collection=req.collection,
        enable_rerank=not fast_mode,
        enable_image_extraction=not fast_mode,
        max_search_queries=2 if fast_mode else None,
    )
    writer_reviewer_agent = WriterReviewerAgent(
        llm=llm,
        template_manager=TemplateManager(),
        trace=trace,
        request=req,
        resolved_template_type=resolved_template_type,
        build_default_prompt=build_lesson_plan_prompt,
        build_fallback_prompt=build_lesson_plan_prompt_fallback,
        integrate_images=integrate_images_into_markdown,
    )
    orchestrator = LessonOrchestrator(
        planner_agent=planner_agent,
        query_agent=query_agent,
        retriever_agent=retriever_agent,
        writer_reviewer_agent=writer_reviewer_agent,
        conversation_agent=conversation_agent,
        trace=trace,
        progress_callback=progress_callback,
    )
    trace.metadata["fast_mode"] = fast_mode
    if fast_mode:
        trace.record_stage(
            "lesson_fast_mode",
            {
                "enabled": True,
                "planner_llm_disabled": True,
                "rerank_disabled": True,
                "image_extraction_disabled": True,
                "top_k": top_k,
                "max_search_queries": 2,
            },
        )

    orchestration_output = orchestrator.run(
        topic=req.topic,
        template_category=req.template_category,
        conversation_state=conversation_state,
    )
    logger.info(
        "lesson_plan.orchestration_done elapsed_ms=%.1f topic=%s",
        (time.monotonic() - started) * 1000,
        req.topic,
    )
    if progress_callback is not None:
        progress_callback(
            "orchestration_done",
            {
                "elapsed_ms": (time.monotonic() - started) * 1000,
                "topic": req.topic,
            },
        )
    execution_plan = dict(orchestration_output["execution_plan"])
    query_plan = dict(orchestration_output["query_plan"])
    relevant_results = list(orchestration_output["results"] or [])
    lesson_plan_content = str(orchestration_output["lesson_plan_content"] or "")
    image_resources = list(orchestration_output["image_resources"] or [])
    existing_image_count = len(image_resources)
    generated_visual = maybe_generate_visual_asset(
        topic=req.topic,
        notes=req.notes,
        template_category=req.template_category,
        existing_images=image_resources,
        llm=llm,
    )
    if generated_visual is not None:
        image_resources.append(generated_visual)
        lesson_plan_content = integrate_images_into_markdown(
            lesson_plan_content,
            [generated_visual],
            start_index=existing_image_count + 1,
            placement_strategy="append",
        )
    if image_resources:
        lesson_plan_content = renumber_image_references(lesson_plan_content)
    subject = orchestration_output.get("subject")
    review_report_payload = orchestration_output.get("review_report")
    generation_metadata = dict(orchestration_output.get("generation_metadata") or {})
    finalized_conversation = orchestration_output["conversation_state"]
    citations = [
        Citation(
            source=str(item.get("source") or "unknown"),
            score=float(item.get("display_score", item.get("score")) or 0.0),
            text=str(item.get("text") or ""),
        )
        for item in (orchestration_output.get("citations") or [])
    ]

    if looks_like_lesson_refusal(lesson_plan_content):
        recovery_messages = [
            Message(
                role="system",
                content=(
                    "你是一名经验丰富的一线教师与教研组长。请直接基于主题与通用学科知识生成完整教案，"
                    "不要输出拒绝、资料不足、要求补充材料等语句。"
                ),
            ),
            Message(role="user", content=f"请为主题“{req.topic}”生成完整成稿。"),
        ]
        lesson_plan_content = llm.chat(recovery_messages).content
        trace.record_stage(
            "lesson_agent_final_refusal_recovery",
            {"applied": True},
        )

    trace.metadata["topic"] = req.topic
    trace.metadata["collection"] = req.collection
    trace.metadata["model"] = req.model or settings.llm.model
    trace.metadata["include_background"] = req.include_background
    trace.metadata["include_facts"] = req.include_facts
    trace.metadata["include_examples"] = req.include_examples
    trace.metadata["template_category"] = req.template_category
    trace.metadata["session_id"] = finalized_conversation.session_id
    trace.metadata["agent_protocol"] = "lesson_agent_msg_v1"
    trace.metadata["query_plan"] = dict(query_plan)
    trace.metadata["execution_plan"] = dict(execution_plan)
    trace.metadata["fast_mode"] = fast_mode

    if isinstance(review_report_payload, dict):
        review_must_fix = list(review_report_payload.get("must_fix") or [])
    else:
        review_must_fix = list(getattr(review_report_payload, "must_fix", []) or [])

    trace.record_stage("lesson_agent_complete", {
        "model": req.model or settings.llm.model,
        "has_context": len(relevant_results) > 0,
        "image_count": len(image_resources),
        "subject": subject,
        "template_type": resolved_template_type,
        "planning_mode": execution_plan.get("generation_mode"),
        "review_must_fix_count": len(review_must_fix),
        "session_id": finalized_conversation.session_id,
    })

    trace.finish()
    from src.core.trace.trace_collector import TraceCollector
    TraceCollector().collect(trace)

    history_records: List[Dict[str, Any]] = []
    try:
        lesson_preview = re.sub(r"\s+", " ", lesson_plan_content or "").strip()[:180]
        request.app.state.history_storage.add_record(
            session_id=finalized_conversation.session_id,
            topic=req.topic,
            notes=req.notes,
            template_category=req.template_category,
            template_label=get_template_label_by_category(req.template_category),
            subject=subject,
            created_at=finalized_conversation.updated_at,
            conversation_state=asdict(finalized_conversation),
            lesson_preview=lesson_preview,
            lesson_content=lesson_plan_content,
            planning_mode=execution_plan.get("generation_mode"),
            used_autonomous_fallback=bool(
                generation_metadata.get("forced_autonomous_retry_after_refusal")
                or generation_metadata.get("forced_autonomous_retry_after_review")
                or execution_plan.get("generation_mode") == "autonomous"
            ),
        )
        history_records = request.app.state.history_storage.list_records(
            limit=8,
            session_id=finalized_conversation.session_id,
        )
    except Exception:
        logger.exception("Failed to persist lesson history")

    return LessonPlanResponse(
        topic=req.topic,
        subject=subject,
        lesson_content=lesson_plan_content,
        additional_resources=citations,
        image_resources=image_resources,
        review_report=to_review_report_response(review_report_payload),
        conversation_state=asdict(finalized_conversation),
        history_records=history_records,
        execution_plan=dict(execution_plan),
        planning_mode=execution_plan.get("generation_mode"),
        used_autonomous_fallback=bool(
            generation_metadata.get("forced_autonomous_retry_after_refusal")
            or generation_metadata.get("forced_autonomous_retry_after_review")
            or execution_plan.get("generation_mode") == "autonomous"
        ),
    )
