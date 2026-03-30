"""Typed tools for lesson workflow Phase-1 runtime integration."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .lesson_agent import LessonAgent


class PlanLessonTaskRequest(BaseModel):
    topic: str
    template_category: Optional[str] = None
    conversation_state: Optional[Dict[str, Any]] = None


class PlanLessonTaskResponse(BaseModel):
    execution_plan: Dict[str, Any]
    query_plan: Dict[str, Any]


class SearchTextChunksRequest(BaseModel):
    query_plan: Dict[str, Any]
    top_k: int = 15


class SearchTextChunksResponse(BaseModel):
    results: List[Any] = Field(default_factory=list)


class GenerateLessonDraftRequest(BaseModel):
    topic: str
    results: List[Any] = Field(default_factory=list)
    image_resources: List[Any] = Field(default_factory=list)
    citations: List[Any] = Field(default_factory=list)
    query_plan: Dict[str, Any]
    execution_plan: Dict[str, Any]
    conversation_state: Dict[str, Any]
    resolved_template_type: Optional[str] = None


class GenerateLessonDraftResponse(BaseModel):
    lesson_plan_content: str
    subject: Optional[str] = None
    review_report: Optional[Dict[str, Any]] = None
    review_notes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LessonToolbox:
    """Tool handlers backed by existing agent components."""

    def __init__(
        self,
        *,
        planner_agent: Any,
        query_agent: Any,
        conversation_state: Any,
        hybrid_search: Any,
        reranker: Any,
        trace: Any,
        llm: Any,
        template_manager: Any,
        request: Any,
        build_default_prompt: Any,
        build_fallback_prompt: Any,
        integrate_images: Any,
    ) -> None:
        self.planner_agent = planner_agent
        self.query_agent = query_agent
        self.conversation_state = conversation_state
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.trace = trace
        self.llm = llm
        self.template_manager = template_manager
        self.request = request
        self.build_default_prompt = build_default_prompt
        self.build_fallback_prompt = build_fallback_prompt
        self.integrate_images = integrate_images
        self.execution_plan: Optional[Any] = None
        self.query_plan: Optional[Any] = None

    def plan_lesson_task(self, req: PlanLessonTaskRequest) -> PlanLessonTaskResponse:
        execution_plan = self.planner_agent.plan(
            topic=req.topic,
            template_category=req.template_category,
            conversation_state=self.conversation_state,
        )
        self.execution_plan = execution_plan
        query_plan = self.query_agent.build_plan(
            topic=req.topic,
            template_category=req.template_category,
            conversation_state=self.conversation_state,
            execution_plan=execution_plan,
        )
        self.query_plan = query_plan
        return PlanLessonTaskResponse(
            execution_plan=asdict(execution_plan),
            query_plan=asdict(query_plan),
        )

    def search_text_chunks(self, req: SearchTextChunksRequest) -> SearchTextChunksResponse:
        query_plan = req.query_plan
        search_queries = list(query_plan.get("search_queries") or [])
        if not search_queries:
            search_queries = [str(query_plan.get("user_query") or "")]
        top_k = max(1, int(req.top_k))
        per_query_top_k = max(6, min(top_k, (top_k // max(1, len(search_queries))) + 3))

        results: List[Any] = []
        seen_result_keys = set()
        for search_query in search_queries:
            hybrid_result = self.hybrid_search.search(
                query=search_query,
                top_k=per_query_top_k,
                filters=None,
                trace=self.trace,
            )
            current_results = hybrid_result if not hasattr(hybrid_result, "results") else hybrid_result.results
            for item in current_results:
                metadata = item.metadata or {}
                key = (
                    metadata.get("chunk_id")
                    or metadata.get("id")
                    or f"{metadata.get('doc_hash')}::{metadata.get('page_num')}::{(item.text or '')[:80]}"
                )
                if key in seen_result_keys:
                    continue
                seen_result_keys.add(key)
                results.append(item)

        results = results[:top_k]
        if results and self.reranker.is_enabled:
            rerank_result = self.reranker.rerank(
                query=str(query_plan.get("user_query") or ""),
                results=results,
                top_k=top_k,
                trace=self.trace,
            )
            results = rerank_result.results

        return SearchTextChunksResponse(results=results)

    def generate_lesson_draft(self, req: GenerateLessonDraftRequest) -> GenerateLessonDraftResponse:
        execution_plan = self.execution_plan
        query_plan = self.query_plan
        if execution_plan is None:
            raise ValueError("Missing execution_plan in toolbox runtime state")
        if query_plan is None:
            raise ValueError("Missing query_plan in toolbox runtime state")
        resolved_template_type = req.resolved_template_type
        agent = LessonAgent(
            llm=self.llm,
            template_manager=self.template_manager,
            trace=self.trace,
            request=self.request,
            resolved_template_type=resolved_template_type,
            build_default_prompt=self.build_default_prompt,
            build_fallback_prompt=self.build_fallback_prompt,
            integrate_images=self.integrate_images,
        )
        agent_state = agent.run(
            topic=req.topic,
            results=req.results,
            image_resources=req.image_resources,
            citations=req.citations,
            query_plan=query_plan,
            execution_plan=execution_plan,
            conversation_state=self.conversation_state,
        )
        return GenerateLessonDraftResponse(
            lesson_plan_content=agent_state.final_content or agent_state.draft_content or "",
            subject=agent_state.subject,
            review_report=asdict(agent_state.review_report) if agent_state.review_report else None,
            review_notes=list(agent_state.review_notes or []),
            metadata=dict(agent_state.metadata or {}),
        )
