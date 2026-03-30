"""Query planning agent for better lesson retrieval."""

from __future__ import annotations

from typing import List, Optional

from .models import ConversationState, QueryPlan
from .planning_models import ExecutionPlan


class QueryAgent:
    """Builds retrieval queries from topic, template, and recent preferences."""

    def build_plan(
        self,
        topic: str,
        template_category: Optional[str],
        conversation_state: Optional[ConversationState] = None,
        execution_plan: Optional[ExecutionPlan] = None,
    ) -> QueryPlan:
        category = template_category or "comprehensive"
        image_focus = category == "comprehensive"
        if execution_plan is not None:
            image_focus = bool(execution_plan.need_images)
        search_queries: List[str] = [topic]
        image_queries: List[str] = []

        base_suffix = "背景 原理 应用 教学"
        if category == "guide":
            search_queries.append(f"{topic} 学习目标 重难点 基础练习 拓展训练 课堂检测")
            search_queries.append(f"{topic} 教材知识点 例题 任务单")
        else:
            search_queries.append(f"{topic} {base_suffix} 课堂活动 案例")
            search_queries.append(f"{topic} 结构图 流程图 示意图 实验结果 图表")
            search_queries.append(f"{topic} 图片 图示 图解 模型结构 网络结构")
            image_queries.append(f"{topic} 结构图 流程图 示意图")
            image_queries.append(f"{topic} 实验结果 图表 对比图")
            image_queries.append(f"{topic} 图解 模型结构 特征图")

        if conversation_state and conversation_state.latest_subject:
            search_queries.append(f"{conversation_state.latest_subject} {topic} 核心概念")
        if execution_plan and execution_plan.subject_guess:
            search_queries.append(f"{execution_plan.subject_guess} {topic} 核心概念")
        if execution_plan and execution_plan.query_focus:
            search_queries.append(f"{topic} {' '.join(execution_plan.query_focus[:5])}")

        # Preserve order while removing duplicates/empties.
        dedup_search = list(dict.fromkeys(query.strip() for query in search_queries if query and query.strip()))
        dedup_image = list(dict.fromkeys(query.strip() for query in image_queries if query and query.strip()))

        return QueryPlan(
            user_query=topic,
            search_queries=dedup_search[:4],
            image_queries=dedup_image[:3],
            intent="lesson_generation" if category == "comprehensive" else "guide_generation",
            image_focus=image_focus,
            reasoning=(
                "Use template-aware + planner-aware query expansion and keep the plan lightweight for stateless deployment."
            ),
        )
