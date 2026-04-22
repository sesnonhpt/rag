"""Three-agent orchestrator for lesson generation."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import time
from typing import Any, Callable, Dict, Optional

from src.observability.logger import get_logger
from src.observability.user_action_tracker import get_tracker

from .agent_protocol import AgentMessage
from .tools.base import ToolExecutor

logger = get_logger(__name__)


class LessonOrchestrator:
    """Orchestrate PlannerAgent -> RetrieverAgent -> WriterReviewerAgent."""

    def __init__(
        self,
        *,
        planner_agent: Any,
        query_agent: Any,
        retriever_agent: Any,
        writer_reviewer_agent: Any,
        conversation_agent: Any,
        trace: Any,
        tool_executor: Optional[ToolExecutor] = None,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.planner_agent = planner_agent
        self.query_agent = query_agent
        self.retriever_agent = retriever_agent
        self.writer_reviewer_agent = writer_reviewer_agent
        self.conversation_agent = conversation_agent
        self.trace = trace
        self.tool_executor = tool_executor
        self.progress_callback = progress_callback

        # Inject live tool schemas into ToolPlanner so LLM knows what's available
        if tool_executor is not None and hasattr(planner_agent, "tool_planner"):
            planner_agent.tool_planner.tool_schemas = tool_executor.get_tool_schemas()

    def run(
        self,
        *,
        topic: str,
        template_category: Optional[str],
        conversation_state: Any,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        overall_started = time.monotonic()
        
        # 开始追踪用户行为
        tracker = get_tracker()
        action_id = tracker.start_action(
            action_type="generate_lesson",
            request_data={
                "topic": topic,
                "template_category": template_category,
                "notes": notes,
            },
            session_id=getattr(conversation_state, "session_id", None),
        )
        
        message = AgentMessage(
            goal=f"生成主题“{topic}”的高质量教学内容",
            context={
                "topic": topic,
                "template_category": template_category,
                "conversation_state": asdict(conversation_state),
                "action_id": action_id,
            },
            constraints=[
                {"key": "template_category", "value": template_category or "comprehensive", "source": "user"},
            ],
            next_action="plan",
        )
        self._record_waterfall("orchestrator_start", message, output=None)
        self._emit_progress(
            "started",
            {
                "topic": topic,
                "template_category": template_category or "comprehensive",
                "session_id": getattr(conversation_state, "session_id", None),
            },
        )
        logger.info(
            "lesson_orchestrator.start topic=%s template_category=%s session_id=%s",
            topic,
            template_category or "comprehensive",
            getattr(conversation_state, "session_id", None),
        )

        planner_started = time.monotonic()
        
        # 追踪 PlannerAgent 执行
        planner_execution_id = tracker.start_agent(
            action_id=action_id,
            agent_name="PlannerAgent",
            input_data={"topic": topic, "template_category": template_category},
        )
        
        execution_plan = self.planner_agent.plan(
            topic=topic,
            template_category=template_category,
            conversation_state=conversation_state,
            notes=notes,
        )
        query_plan = self.query_agent.build_plan(
            topic=topic,
            template_category=template_category,
            conversation_state=conversation_state,
            execution_plan=execution_plan,
        )
        message.artifacts["execution_plan"] = asdict(execution_plan)
        message.artifacts["query_plan"] = asdict(query_plan)
        message.next_action = "retrieve"
        
        # 完成 PlannerAgent 追踪
        tracker.complete_agent(
            action_id=action_id,
            execution_id=planner_execution_id,
            output_data={
                "plan_version": execution_plan.plan_version,
                "generation_mode": execution_plan.generation_mode,
                "need_images": execution_plan.need_images,
                "tool_calls": len(execution_plan.tool_calls),
            },
        )
        
        self._record_waterfall(
            "planner_agent",
            message,
            output={
                "plan_version": execution_plan.plan_version,
                "generation_mode": execution_plan.generation_mode,
                "need_images": execution_plan.need_images,
                "search_query_count": len(query_plan.search_queries or []),
            },
            elapsed_ms=(time.monotonic() - planner_started) * 1000,
        )
        logger.info(
            "lesson_orchestrator.planner_done elapsed_ms=%.1f generation_mode=%s need_images=%s search_queries=%s",
            (time.monotonic() - planner_started) * 1000,
            execution_plan.generation_mode,
            execution_plan.need_images,
            len(query_plan.search_queries or []),
        )
        self._emit_progress(
            "planner_done",
            {
                "elapsed_ms": (time.monotonic() - planner_started) * 1000,
                "generation_mode": execution_plan.generation_mode,
                "need_images": execution_plan.need_images,
                "search_query_count": len(query_plan.search_queries or []),
            },
        )

        self.conversation_agent.apply_plan_to_state(conversation_state, message.artifacts["execution_plan"])

        # Execute tools if available
        tool_results = []
        if self.tool_executor and execution_plan.tool_calls:
            # Filter out deferred tools
            immediate_tools = [
                call for call in execution_plan.tool_calls
                if not call.get("deferred", False)
            ]
            
            if immediate_tools:
                tool_started = time.monotonic()
                try:
                    tool_results = asyncio.run(
                        self.tool_executor.execute_parallel(immediate_tools)
                    )
                    
                    # 追踪每个工具执行
                    for tool_result in tool_results:
                        tracker.record_tool(
                            action_id=action_id,
                            execution_id=planner_execution_id,
                            tool_name=tool_result.tool_name,
                            params=immediate_tools[[t["tool_name"] for t in immediate_tools].index(tool_result.tool_name)].get("params", {}),
                            result=tool_result.to_dict(),
                            elapsed_ms=tool_result.elapsed_ms,
                            status="completed" if tool_result.success else "failed",
                            error=tool_result.error,
                            degraded=tool_result.metadata.get("degraded", False),
                        )
                    
                    self._record_waterfall(
                        "tool_executor",
                        message,
                        output={
                            "tool_count": len(tool_results),
                            "success_count": sum(1 for r in tool_results if r.success),
                            "tools": [r.tool_name for r in tool_results],
                        },
                        elapsed_ms=(time.monotonic() - tool_started) * 1000,
                    )
                    logger.info(
                        "lesson_orchestrator.tools_done elapsed_ms=%.1f tool_count=%d success_count=%d",
                        (time.monotonic() - tool_started) * 1000,
                        len(tool_results),
                        sum(1 for r in tool_results if r.success),
                    )
                    self._emit_progress(
                        "tools_done",
                        {
                            "elapsed_ms": (time.monotonic() - tool_started) * 1000,
                            "tool_count": len(tool_results),
                            "success_count": sum(1 for r in tool_results if r.success),
                        },
                    )
                except Exception as e:
                    logger.error(
                        "lesson_orchestrator.tools_error error=%s",
                        str(e),
                        exc_info=True,
                    )
                    # Continue without tool results (degradation)

        before_retrieval = message.summary()
        retriever_started = time.monotonic()
        message = self.retriever_agent.run(message)
        self._record_waterfall(
            "retriever_agent",
            AgentMessage(
                task_id=message.task_id,
                goal=message.goal,
                context=message.context,
                artifacts={**message.artifacts, "_input_snapshot": before_retrieval},
                constraints=message.constraints,
                next_action=message.next_action,
            ),
            output={
                "raw_result_count": len(message.artifacts.get("raw_results") or []),
                "relevant_result_count": len(message.artifacts.get("relevant_results") or []),
                "citation_count": len(message.artifacts.get("citations") or []),
                "image_count": len(message.artifacts.get("image_resources") or []),
            },
            elapsed_ms=(time.monotonic() - retriever_started) * 1000,
        )
        logger.info(
            "lesson_orchestrator.retriever_done elapsed_ms=%.1f raw_results=%s relevant_results=%s citations=%s images=%s",
            (time.monotonic() - retriever_started) * 1000,
            len(message.artifacts.get("raw_results") or []),
            len(message.artifacts.get("relevant_results") or []),
            len(message.artifacts.get("citations") or []),
            len(message.artifacts.get("image_resources") or []),
        )
        self._emit_progress(
            "retriever_done",
            {
                "elapsed_ms": (time.monotonic() - retriever_started) * 1000,
                "raw_result_count": len(message.artifacts.get("raw_results") or []),
                "relevant_result_count": len(message.artifacts.get("relevant_results") or []),
                "citation_count": len(message.artifacts.get("citations") or []),
                "image_count": len(message.artifacts.get("image_resources") or []),
            },
        )

        writer_started = time.monotonic()
        writer_output = self.writer_reviewer_agent.run(
            topic=topic,
            results=message.artifacts.get("relevant_results") or [],
            image_resources=message.artifacts.get("image_resources") or [],
            citations=message.artifacts.get("citations") or [],
            query_plan=query_plan,
            execution_plan=execution_plan,
            conversation_state=conversation_state,
            tool_results=tool_results,  # Pass tool results to writer
        )
        review_report = writer_output.get("review_report")
        if isinstance(review_report, dict):
            must_fix_count = len(review_report.get("must_fix") or [])
        else:
            must_fix_count = len(getattr(review_report, "must_fix", []) or [])
        self._record_waterfall(
            "writer_reviewer_agent",
            message,
            output={
                "has_content": bool(writer_output.get("lesson_plan_content")),
                "subject": writer_output.get("subject"),
                "review_note_count": len(writer_output.get("review_notes") or []),
                "must_fix_count": must_fix_count,
            },
            elapsed_ms=(time.monotonic() - writer_started) * 1000,
        )
        logger.info(
            "lesson_orchestrator.writer_done elapsed_ms=%.1f has_content=%s subject=%s review_notes=%s must_fix=%s",
            (time.monotonic() - writer_started) * 1000,
            bool(writer_output.get("lesson_plan_content")),
            writer_output.get("subject"),
            len(writer_output.get("review_notes") or []),
            must_fix_count,
        )
        self._emit_progress(
            "writer_done",
            {
                "elapsed_ms": (time.monotonic() - writer_started) * 1000,
                "has_content": bool(writer_output.get("lesson_plan_content")),
                "subject": writer_output.get("subject"),
                "review_note_count": len(writer_output.get("review_notes") or []),
                "must_fix_count": must_fix_count,
            },
        )

        finalized_conversation = self.conversation_agent.finalize_state(
            conversation_state,
            subject=writer_output.get("subject"),
            review_notes=writer_output.get("review_notes") or [],
            query_plan=query_plan,
            execution_plan=execution_plan,
        )

        total_elapsed_ms = (time.monotonic() - overall_started) * 1000
        
        # 完成用户行为追踪
        tracker.complete_action(action_id=action_id, final_status="completed")
        
        logger.info(
            "lesson_orchestrator.complete elapsed_ms=%.1f topic=%s session_id=%s action_id=%s",
            total_elapsed_ms,
            topic,
            getattr(finalized_conversation, "session_id", None),
            action_id,
        )
        self._emit_progress(
            "completed",
            {
                "elapsed_ms": total_elapsed_ms,
                "topic": topic,
                "session_id": getattr(finalized_conversation, "session_id", None),
            },
        )

        return {
            "execution_plan": asdict(execution_plan),
            "query_plan": asdict(query_plan),
            "conversation_state": finalized_conversation,
            "results": message.artifacts.get("relevant_results") or [],
            "raw_results": message.artifacts.get("raw_results") or [],
            "citations": message.artifacts.get("citations") or [],
            "image_resources": message.artifacts.get("image_resources") or [],
            "lesson_plan_content": writer_output.get("lesson_plan_content") or "",
            "subject": writer_output.get("subject"),
            "review_report": writer_output.get("review_report"),
            "review_notes": writer_output.get("review_notes") or [],
            "generation_metadata": writer_output.get("metadata") or {},
            "tool_results": [r.to_dict() for r in tool_results],  # Include tool results in output
        }

    def _record_waterfall(
        self,
        agent_name: str,
        message: AgentMessage,
        output: Optional[Dict[str, Any]],
        elapsed_ms: Optional[float] = None,
    ) -> None:
        data: Dict[str, Any] = {
            "agent": agent_name,
            "message": message.summary(),
        }
        if output is not None:
            data["output"] = output
        self.trace.record_stage(f"agent_waterfall_{agent_name}", data, elapsed_ms=elapsed_ms)

    def _emit_progress(self, stage: str, payload: Dict[str, Any]) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(stage, payload)
