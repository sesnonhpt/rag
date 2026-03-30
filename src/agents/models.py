"""State models for lesson-generation agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .planning_models import ExecutionPlan, PlanConstraint

@dataclass
class LessonAgentAssets:
    """Assets retrieved for lesson generation."""

    text_results: List[Any] = field(default_factory=list)
    image_resources: List[Any] = field(default_factory=list)
    citations: List[Any] = field(default_factory=list)


@dataclass
class LessonReviewReport:
    """Structured review result for a lesson draft."""

    realism_score: int = 0
    pedagogy_score: int = 0
    structure_score: int = 0
    multimodal_score: int = 0
    strengths: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    must_fix: List[str] = field(default_factory=list)


@dataclass
class QueryPlan:
    """Structured query plan produced before retrieval."""

    user_query: str
    search_queries: List[str] = field(default_factory=list)
    image_queries: List[str] = field(default_factory=list)
    intent: str = "lesson_generation"
    image_focus: bool = False
    reasoning: str = ""


@dataclass
class ConversationState:
    """Render-friendly stateless conversation state.

    The client stores and resubmits this state, so the server remains stateless.
    """

    session_id: str = field(default_factory=lambda: uuid4().hex)
    current_topic: Optional[str] = None
    template_category: Optional[str] = None
    recent_topics: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    latest_feedback: List[str] = field(default_factory=list)
    latest_subject: Optional[str] = None
    constraints: List[PlanConstraint] = field(default_factory=list)
    task_memory: Dict[str, Any] = field(default_factory=dict)
    last_plan: Optional[Dict[str, Any]] = None
    last_query_plan: Optional[Dict[str, Any]] = None
    plan_version: str = "planner_v1"
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class LessonAgentState:
    """Mutable state for the lesson-generation workflow."""

    topic: str
    template_category: Optional[str] = None
    template_type: Optional[str] = None
    subject: Optional[str] = None
    assets: LessonAgentAssets = field(default_factory=LessonAgentAssets)
    draft_content: Optional[str] = None
    final_content: Optional[str] = None
    review_notes: List[str] = field(default_factory=list)
    review_report: Optional[LessonReviewReport] = None
    query_plan: Optional[QueryPlan] = None
    execution_plan: Optional[ExecutionPlan] = None
    conversation_state: Optional[ConversationState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
