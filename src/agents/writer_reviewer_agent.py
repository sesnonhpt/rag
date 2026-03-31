"""Writer/Reviewer agent for lesson workflow."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .lesson_agent import LessonAgent


class WriterReviewerAgent:
    """Generate and polish lesson draft from retrieved artifacts."""

    def __init__(
        self,
        *,
        llm: Any,
        template_manager: Any,
        trace: Any,
        request: Any,
        resolved_template_type: Optional[str],
        build_default_prompt: Any,
        build_fallback_prompt: Any,
        integrate_images: Any,
    ) -> None:
        self.agent = LessonAgent(
            llm=llm,
            template_manager=template_manager,
            trace=trace,
            request=request,
            resolved_template_type=resolved_template_type,
            build_default_prompt=build_default_prompt,
            build_fallback_prompt=build_fallback_prompt,
            integrate_images=integrate_images,
        )

    def run(
        self,
        *,
        topic: str,
        results: Any,
        image_resources: Any,
        citations: Any,
        query_plan: Any,
        execution_plan: Any,
        conversation_state: Any,
    ) -> Dict[str, Any]:
        state = self.agent.run(
            topic=topic,
            results=results,
            image_resources=image_resources,
            citations=citations,
            query_plan=query_plan,
            execution_plan=execution_plan,
            conversation_state=conversation_state,
        )
        return {
            "lesson_plan_content": state.final_content or state.draft_content or "",
            "subject": state.subject,
            "review_report": state.review_report,
            "review_notes": list(state.review_notes or []),
            "metadata": dict(state.metadata or {}),
        }
