"""Conversation state manager for lesson-generation flows.

Designed for Render-like stateless deployment:
the client sends back the lightweight conversation state on each request.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .models import ConversationState, QueryPlan
from .planning_models import ExecutionPlan, PlanConstraint


class ConversationAgent:
    """Maintains lightweight structured history across requests."""

    def prepare_state(
        self,
        request: Any,
        previous_state: Optional[Dict[str, Any]] = None,
    ) -> ConversationState:
        state = self._coerce_state(previous_state)
        state.current_topic = getattr(request, "topic", None) or state.current_topic
        state.template_category = getattr(request, "template_category", None) or state.template_category

        if state.current_topic:
            recent_topics = [state.current_topic] + [topic for topic in state.recent_topics if topic != state.current_topic]
            state.recent_topics = recent_topics[:5]

        preferences = dict(state.user_preferences or {})
        preferences.setdefault("prefer_visual_lesson", True)
        preferences.setdefault("hide_absolute_paths", True)
        preferences.setdefault("allow_teaching_expansion", True)
        preferences.setdefault("filter_invalid_images", True)
        if getattr(request, "template_category", None):
            preferences["last_selected_template"] = request.template_category
        state.user_preferences = preferences
        state.task_memory["active_template_category"] = state.template_category
        state.task_memory["active_topic"] = state.current_topic
        state.updated_at = datetime.now(timezone.utc).isoformat()
        return state

    def finalize_state(
        self,
        state: ConversationState,
        subject: Optional[str],
        review_notes: Optional[list[str]],
        query_plan: Optional[QueryPlan],
        execution_plan: Optional[ExecutionPlan] = None,
    ) -> ConversationState:
        if subject:
            state.latest_subject = subject
            state.task_memory["latest_subject"] = subject
        if review_notes:
            state.latest_feedback = list(review_notes)[:5]
            state.task_memory["latest_feedback"] = list(review_notes)[:5]
        if query_plan:
            state.last_query_plan = asdict(query_plan)
        if execution_plan:
            state.last_plan = self._plan_to_dict(execution_plan)
            state.plan_version = str(self._plan_value(execution_plan, "plan_version", "planner_v1"))
            self.apply_plan_to_state(state, execution_plan)
        state.updated_at = datetime.now(timezone.utc).isoformat()
        return state

    def merge_constraints(
        self,
        state: ConversationState,
        plan_constraints: list[PlanConstraint],
    ) -> None:
        existing = {(constraint.key, str(constraint.value)) for constraint in state.constraints}
        for constraint in plan_constraints:
            key = getattr(constraint, "key", None)
            value = getattr(constraint, "value", None)
            source = getattr(constraint, "source", "planner")
            if key is None:
                continue
            marker = (str(key), str(value))
            if marker in existing:
                continue
            state.constraints.append(PlanConstraint(key=str(key), value=value, source=str(source)))
            existing.add(marker)
        state.constraints = state.constraints[-20:]

    def apply_plan_to_state(self, state: ConversationState, execution_plan: ExecutionPlan) -> None:
        state.task_memory["generation_mode"] = self._plan_value(execution_plan, "generation_mode", "context_first")
        state.task_memory["need_images"] = bool(self._plan_value(execution_plan, "need_images", False))
        state.task_memory["query_focus"] = list(self._plan_value(execution_plan, "query_focus", []))
        subject_guess = self._plan_value(execution_plan, "subject_guess", None)
        if subject_guess and not state.latest_subject:
            state.task_memory["subject_guess"] = subject_guess
        raw_constraints = list(self._plan_value(execution_plan, "constraints", []))
        constraints: list[PlanConstraint] = []
        for item in raw_constraints:
            if isinstance(item, PlanConstraint):
                constraints.append(item)
                continue
            if isinstance(item, dict):
                key = str(item.get("key") or "").strip()
                if not key:
                    continue
                constraints.append(
                    PlanConstraint(
                        key=key,
                        value=item.get("value"),
                        source=str(item.get("source") or "planner"),
                    )
                )
        self.merge_constraints(state, constraints)

    def _coerce_state(self, payload: Optional[Dict[str, Any]]) -> ConversationState:
        if not payload:
            return ConversationState()

        raw_constraints = list(payload.get("constraints") or [])
        constraints: list[PlanConstraint] = []
        for item in raw_constraints:
            if not isinstance(item, dict):
                continue
            constraints.append(
                PlanConstraint(
                    key=str(item.get("key") or ""),
                    value=item.get("value"),
                    source=str(item.get("source") or "user"),
                )
            )

        return ConversationState(
            session_id=str(payload.get("session_id") or ConversationState().session_id),
            current_topic=payload.get("current_topic"),
            template_category=payload.get("template_category"),
            recent_topics=list(payload.get("recent_topics") or []),
            user_preferences=dict(payload.get("user_preferences") or {}),
            latest_feedback=list(payload.get("latest_feedback") or []),
            latest_subject=payload.get("latest_subject"),
            constraints=constraints,
            task_memory=dict(payload.get("task_memory") or {}),
            last_plan=dict(payload.get("last_plan") or {}) or None,
            last_query_plan=dict(payload.get("last_query_plan") or {}) or None,
            plan_version=str(payload.get("plan_version") or "planner_v1"),
            updated_at=str(payload.get("updated_at") or datetime.now(timezone.utc).isoformat()),
        )

    @staticmethod
    def _plan_to_dict(execution_plan: Any) -> Dict[str, Any]:
        if isinstance(execution_plan, dict):
            return dict(execution_plan)
        return asdict(execution_plan)

    @staticmethod
    def _plan_value(execution_plan: Any, key: str, default: Any) -> Any:
        if isinstance(execution_plan, dict):
            return execution_plan.get(key, default)
        return getattr(execution_plan, key, default)
