"""Conversation state manager for lesson-generation flows.

Designed for Render-like stateless deployment:
the client sends back the lightweight conversation state on each request.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .models import ConversationState, QueryPlan


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
        state.updated_at = datetime.now(timezone.utc).isoformat()
        return state

    def finalize_state(
        self,
        state: ConversationState,
        subject: Optional[str],
        review_notes: Optional[list[str]],
        query_plan: Optional[QueryPlan],
    ) -> ConversationState:
        if subject:
            state.latest_subject = subject
        if review_notes:
            state.latest_feedback = list(review_notes)[:5]
        if query_plan:
            state.last_query_plan = asdict(query_plan)
        state.updated_at = datetime.now(timezone.utc).isoformat()
        return state

    def _coerce_state(self, payload: Optional[Dict[str, Any]]) -> ConversationState:
        if not payload:
            return ConversationState()

        return ConversationState(
            session_id=str(payload.get("session_id") or ConversationState().session_id),
            current_topic=payload.get("current_topic"),
            template_category=payload.get("template_category"),
            recent_topics=list(payload.get("recent_topics") or []),
            user_preferences=dict(payload.get("user_preferences") or {}),
            latest_feedback=list(payload.get("latest_feedback") or []),
            latest_subject=payload.get("latest_subject"),
            last_query_plan=dict(payload.get("last_query_plan") or {}) or None,
            updated_at=str(payload.get("updated_at") or datetime.now(timezone.utc).isoformat()),
        )
