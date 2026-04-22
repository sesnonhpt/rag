"""Tool planner for determining which tools to use based on topic."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.libs.llm.base_llm import Message
from src.observability.logger import get_logger

logger = get_logger(__name__)

# Default tool schemas used when no ToolExecutor is available
_DEFAULT_TOOL_SCHEMAS = [
    {
        "name": "web_search",
        "description": "搜索互联网获取最新资料、时事信息、前沿技术进展。适用于主题涉及最新动态、当前数据或时效性内容时。",
        "parameters": {
            "query": {"type": "string", "description": "搜索关键词，建议包含学科和主题，如'量子计算 物理 2026'"},
        },
        "required": ["query"],
    },
    {
        "name": "image_retrieval",
        "description": "从图片库中检索与主题相关的图片资源。适用于需要实验装置图、结构图、流程图、几何图形等视觉内容时。",
        "parameters": {
            "description": {"type": "string", "description": "图片描述，包含学科和视觉元素，如'物理 牛顿第二定律 受力分析图'"},
        },
        "required": ["description"],
    },
    {
        "name": "latex_renderer",
        "description": "验证 LaTeX 数学公式语法。适用于学科涉及数学公式、物理方程、化学方程式时。",
        "parameters": {
            "latex_code": {"type": "string", "description": "需要验证的 LaTeX 代码，如 'F = ma' 或 '\\\\frac{d}{dx}'"},
        },
        "required": ["latex_code"],
    },
]


def _build_tool_planner_system_prompt() -> str:
    return (
        "你是教学工具规划器。根据教案主题和学科，决定需要调用哪些工具来增强教案质量。\n"
        "请分析主题特点，选择合适的工具并生成调用参数。\n"
        "仅输出 JSON 数组，每个元素包含 tool_name、params、reason 字段。\n"
        "如果不需要任何工具，输出空数组 []。"
    )


def _build_tool_planner_user_prompt(
    topic: str,
    subject: Optional[str],
    template_category: str,
    tool_schemas: List[Dict[str, Any]],
    notes: Optional[str] = None,
) -> str:
    tools_desc = "\n".join(
        f"- {t['name']}: {t['description']}" for t in tool_schemas
    )
    notes_line = f"教师备注：{notes}\n" if notes else ""
    return (
        f"主题：{topic}\n"
        f"学科：{subject or '未知'}\n"
        f"模板类型：{template_category}\n"
        f"{notes_line}"
        f"\n可用工具：\n{tools_desc}\n\n"
        "请决定需要调用哪些工具，输出格式：\n"
        '[\n'
        '  {"tool_name": "工具名", "params": {"参数名": "参数值"}, "reason": "使用原因"},\n'
        '  ...\n'
        ']'
    )


class ToolPlanner:
    """Determine which tools should be used for a given topic.

    Uses LLM-based decision making when an LLM is available,
    falls back to heuristic rules otherwise.
    """

    def __init__(self, llm: Optional[Any] = None, tool_schemas: Optional[List[Dict[str, Any]]] = None):
        self.llm = llm
        self.tool_schemas = tool_schemas or _DEFAULT_TOOL_SCHEMAS

    def plan_tools(
        self,
        *,
        topic: str,
        subject: Optional[str] = None,
        template_category: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Plan which tools to use. LLM-first, heuristic fallback."""
        category = template_category or "comprehensive"

        if self.llm is not None:
            llm_result = self._llm_plan_tools(topic=topic, subject=subject, template_category=category, notes=notes)
            if llm_result is not None:
                logger.info(
                    "tool_planner.llm_plan topic=%s subject=%s notes=%s tool_count=%d tools=%s",
                    topic[:50],
                    subject,
                    (notes or "")[:30],
                    len(llm_result),
                    [t["tool_name"] for t in llm_result],
                )
                return llm_result

        # Fallback to heuristic
        result = self._heuristic_plan_tools(topic=topic, subject=subject, template_category=category, notes=notes)
        logger.info(
            "tool_planner.heuristic_plan topic=%s subject=%s notes=%s tool_count=%d tools=%s",
            topic[:50],
            subject,
            (notes or "")[:30],
            len(result),
            [t["tool_name"] for t in result],
        )
        return result

    def _llm_plan_tools(
        self,
        *,
        topic: str,
        subject: Optional[str],
        template_category: str,
        notes: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Use LLM to decide which tools to call."""
        messages = [
            Message(role="system", content=_build_tool_planner_system_prompt()),
            Message(
                role="user",
                content=_build_tool_planner_user_prompt(
                    topic=topic,
                    subject=subject,
                    template_category=template_category,
                    tool_schemas=self.tool_schemas,
                    notes=notes,
                ),
            ),
        ]

        try:
            response = self.llm.chat(messages)
            tool_calls = self._parse_tool_calls(response.content)
            if tool_calls is None:
                logger.warning("tool_planner.llm_parse_failed raw=%s", response.content[:200])
                return None
            return self._validate_tool_calls(tool_calls)
        except Exception as e:
            logger.warning("tool_planner.llm_error error=%s", str(e))
            return None

    def _parse_tool_calls(self, raw: str) -> Optional[List[Dict[str, Any]]]:
        """Parse LLM output into tool call list."""
        text = str(raw or "").strip()

        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()

        # Find JSON array
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None

        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, list) else None
        except Exception:
            return None

    def _validate_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """Validate and normalize tool calls from LLM output."""
        valid_names = {s["name"] for s in self.tool_schemas}
        result = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            tool_name = call.get("tool_name")
            if tool_name not in valid_names:
                logger.warning("tool_planner.unknown_tool tool=%s", tool_name)
                continue
            params = call.get("params", {})
            if not isinstance(params, dict):
                params = {}
            result.append({
                "tool_name": tool_name,
                "params": params,
                "reason": str(call.get("reason", "")),
            })
        return result

    # ── Heuristic fallback ────────────────────────────────────────────────────

    def _heuristic_plan_tools(
        self,
        *,
        topic: str,
        subject: Optional[str],
        template_category: str,
        notes: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Rule-based fallback when LLM is unavailable."""
        tool_calls = []
        combined = f"{topic} {notes or ''}"

        if self._should_use_web_search(combined, subject):
            tool_calls.append({
                "tool_name": "web_search",
                "params": {"query": self._build_search_query(topic, subject, notes)},
                "reason": "获取最新资料和时事信息",
            })

        if self._should_use_images(combined, subject, template_category):
            tool_calls.append({
                "tool_name": "image_retrieval",
                "params": {"description": self._build_image_description(topic, subject)},
                "reason": "检索相关图片资源",
            })

        if self._should_use_latex(combined, subject):
            tool_calls.append({
                "tool_name": "latex_renderer",
                "params": {"latex_code": ""},
                "reason": "验证数学公式语法",
                "deferred": True,
            })

        return tool_calls

    def _should_use_web_search(self, topic: str, subject: Optional[str]) -> bool:
        topic_lower = topic.lower()
        latest_keywords = [
            "最新", "当前", "现在", "今年", "2024", "2025", "2026",
            "进展", "发展", "趋势", "应用", "技术",
            "量子计算", "人工智能", "新能源", "疫情",
            "案例", "实例", "例子", "现实", "实际",
        ]
        return any(keyword in topic_lower for keyword in latest_keywords)

    def _should_use_images(self, topic: str, subject: Optional[str], template_category: str) -> bool:
        if template_category in {"comprehensive", "teaching_design", "ppt"}:
            return True
        topic_lower = topic.lower()
        if subject in {"物理", "化学", "生物", "数学", "地理"}:
            return True
        visual_keywords = ["结构", "图", "实验", "装置", "现象", "流程", "几何", "函数", "曲线", "分子", "细胞", "器官"]
        return any(keyword in topic_lower for keyword in visual_keywords)

    def _should_use_latex(self, topic: str, subject: Optional[str]) -> bool:
        if subject in {"数学", "物理", "化学"}:
            return True
        topic_lower = topic.lower()
        math_keywords = ["公式", "方程", "函数", "定理", "定律", "计算", "推导", "证明", "微积分", "矩阵", "f=ma", "e=mc", "牛顿", "欧拉"]
        return any(keyword in topic_lower for keyword in math_keywords)

    def _build_search_query(self, topic: str, subject: Optional[str], notes: Optional[str] = None) -> str:
        parts = [subject, topic, notes] if subject else [topic, notes]
        query = " ".join(p for p in parts if p)
        return f"{query} 2026"[:200]

    def _build_image_description(self, topic: str, subject: Optional[str]) -> str:
        desc = f"{subject} {topic}" if subject else topic
        return desc[:200]
