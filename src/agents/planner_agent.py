"""Planner agent for lesson generation."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from src.libs.llm.base_llm import Message

from .planning_models import ExecutionPlan, PlanConstraint
from .prompts.planner_prompts import build_planner_system_prompt, build_planner_user_payload


class PlannerAgent:
    """Build a structured execution plan before retrieval and writing."""

    def __init__(self, llm: Optional[Any] = None) -> None:
        self.llm = llm

    def plan(
        self,
        *,
        topic: str,
        template_category: Optional[str],
        conversation_state: Optional[Any] = None,
    ) -> ExecutionPlan:
        category = template_category or "comprehensive"
        heuristic_plan = self._heuristic_plan(topic=topic, template_category=category, conversation_state=conversation_state)

        if self.llm is None:
            return heuristic_plan

        llm_plan = self._llm_plan(
            topic=topic,
            template_category=category,
            conversation_state=conversation_state,
            fallback=heuristic_plan,
        )
        if llm_plan is None:
            return heuristic_plan
        return llm_plan

    def _heuristic_plan(self, *, topic: str, template_category: str, conversation_state: Optional[Any]) -> ExecutionPlan:
        topic_text = str(topic or "")
        subject_guess = self._guess_subject(topic_text)
        need_images = template_category == "comprehensive"
        generation_mode = "context_first"

        preferences: Dict[str, Any] = {}
        feedback: List[str] = []
        if conversation_state is not None:
            preferences = dict(getattr(conversation_state, "user_preferences", {}) or {})
            feedback = list(getattr(conversation_state, "latest_feedback", []) or [])

        if preferences.get("prefer_visual_lesson") and template_category == "comprehensive":
            need_images = True

        if any("上下文" in str(item) and "不相关" in str(item) for item in feedback):
            generation_mode = "autonomous"

        query_focus = ["核心概念", "课堂活动", "典型案例"]
        if need_images:
            query_focus.extend(["结构图", "流程图", "实验结果图"])
        if subject_guess == "物理":
            query_focus.extend(["实验现象", "规律应用"])
        if subject_guess == "数学":
            query_focus.extend(["例题", "变式训练"])

        constraints: List[PlanConstraint] = [
            PlanConstraint(key="template_category", value=template_category, source="user"),
            PlanConstraint(key="need_images", value=need_images, source="planner"),
            PlanConstraint(key="generation_mode", value=generation_mode, source="planner"),
        ]

        return ExecutionPlan(
            subject_guess=subject_guess,
            grade_hint=self._guess_grade(topic_text),
            lesson_type_hint="new_lesson",
            need_images=need_images,
            query_focus=list(dict.fromkeys(query_focus)),
            generation_mode=generation_mode,
            retry_policy={
                "max_generation_retry": 2,
                "fallback_to_autonomous_on_refusal": True,
            },
            constraints=constraints,
            planner_notes="Heuristic planning based on template, topic, and short-term memory.",
            plan_version="planner_v1",
        )

    def _llm_plan(
        self,
        *,
        topic: str,
        template_category: str,
        conversation_state: Optional[Any],
        fallback: ExecutionPlan,
    ) -> Optional[ExecutionPlan]:
        preferences = {}
        feedback: List[str] = []
        if conversation_state is not None:
            preferences = dict(getattr(conversation_state, "user_preferences", {}) or {})
            feedback = list(getattr(conversation_state, "latest_feedback", []) or [])

        messages = [
            Message(role="system", content=build_planner_system_prompt()),
            Message(
                role="user",
                content=build_planner_user_payload(
                    topic=topic,
                    template_category=template_category,
                    state_preferences=preferences,
                    recent_feedback=feedback,
                ),
            ),
        ]

        try:
            response = self.llm.chat(messages)
            payload = self._parse_json_object(response.content)
            if not payload:
                return None
        except Exception:
            return None

        plan = ExecutionPlan(
            subject_guess=str(payload.get("subject_guess") or fallback.subject_guess),
            grade_hint=str(payload.get("grade_hint") or fallback.grade_hint),
            lesson_type_hint=str(payload.get("lesson_type_hint") or fallback.lesson_type_hint),
            need_images=bool(payload.get("need_images", fallback.need_images)),
            query_focus=self._safe_str_list(payload.get("query_focus")) or fallback.query_focus,
            generation_mode=str(payload.get("generation_mode") or fallback.generation_mode),
            retry_policy=payload.get("retry_policy") if isinstance(payload.get("retry_policy"), dict) else fallback.retry_policy,
            constraints=fallback.constraints,
            planner_notes=str(payload.get("planner_notes") or "LLM planner output normalized by heuristic fallback."),
            plan_version="planner_v1",
        )
        if plan.generation_mode not in {"context_first", "autonomous"}:
            plan.generation_mode = fallback.generation_mode
        return plan

    @staticmethod
    def _guess_subject(topic: str) -> Optional[str]:
        text = topic.lower()
        mapping = [
            ("物理", ["电磁", "力学", "热学", "光学", "法拉第", "楞次"]),
            ("化学", ["化学", "反应", "分子", "离子"]),
            ("生物", ["生物", "细胞", "遗传"]),
            ("数学", ["数学", "函数", "几何", "方程"]),
            ("语文", ["语文", "文言", "古诗", "阅读"]),
            ("英语", ["英语", "grammar", "reading", "vocabulary"]),
            ("历史", ["历史", "朝代", "战争", "史料"]),
            ("地理", ["地理", "气候", "地形", "区域"]),
            ("计算机科学", ["神经网络", "机器学习", "算法", "编程", "python"]),
        ]
        for subject, keywords in mapping:
            if any(keyword.lower() in text for keyword in keywords):
                return subject
        return None

    @staticmethod
    def _guess_grade(topic: str) -> Optional[str]:
        text = topic.lower()
        if any(key in text for key in ["高中", "高一", "高二", "高三"]):
            return "high"
        if any(key in text for key in ["初中", "初一", "初二", "初三"]):
            return "middle"
        if any(key in text for key in ["小学", "三年级", "四年级", "五年级", "六年级"]):
            return "primary"
        return None

    @staticmethod
    def _parse_json_object(raw: str) -> Dict[str, Any]:
        text = str(raw or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _safe_str_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def to_dict(plan: ExecutionPlan) -> Dict[str, Any]:
        return asdict(plan)
