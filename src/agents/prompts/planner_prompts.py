"""Prompt helpers for planner agent."""

from __future__ import annotations

from typing import Any, Dict, List


def build_planner_system_prompt() -> str:
    return (
        "你是教学内容规划器。请根据用户主题和会话偏好，输出结构化执行计划。"
        "规划目标是提高检索匹配和教案可用性，避免无关上下文导致拒绝生成。"
        "仅输出 JSON。"
    )


def build_planner_user_payload(
    topic: str,
    template_category: str,
    state_preferences: Dict[str, Any],
    recent_feedback: List[str],
) -> str:
    return (
        f"主题：{topic}\n"
        f"模板类别：{template_category}\n"
        f"偏好：{state_preferences}\n"
        f"最近反馈：{recent_feedback}\n\n"
        "请给出 execution plan，包括：subject_guess, need_images, query_focus,"
        " generation_mode, retry_policy, planner_notes。"
    )
