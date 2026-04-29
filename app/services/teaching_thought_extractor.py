"""
备课思路提取服务

拆解导学案的教学设计逻辑，帮助老师理解"设计者为什么这样设计"。
数据来源：导学案文本本身（不依赖课堂实录或视频）。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from src.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TeachingThought:
    """备课思路维度"""
    dimension: str
    icon: str
    title: str
    content: str       # 一句话概括
    key_points: List[str]  # 2-4 条要点

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 5 个维度的元数据，顺序固定
DIMENSIONS = [
    ("what_to_teach",    "📚", "教什么"),
    ("how_to_open",      "🚪", "怎么铺垫"),
    ("design_logic",     "🧩", "主线逻辑"),
    ("hard_points",      "⚠️",  "难在哪"),
    ("how_to_close",     "✅", "怎么收束"),
]


class TeachingThoughtExtractor:
    """从导学案文本中拆解教学设计思路"""

    def __init__(self, llm):
        self.llm = llm
        self.llm_timeout_sec = float(os.environ.get("GUIDE_ANALYZER_LLM_TIMEOUT_SEC", "20"))

    async def extract_thoughts(
        self,
        course_text: str,
        subject: str,
        topic: str,
        grade: str = "",
        teacher_name: str = "",
    ) -> List[TeachingThought]:
        logger.info(
            "teaching_thought_extractor.extract_start subject=%s topic=%s",
            subject, topic,
        )

        try:
            import asyncio
            from src.libs.llm.base_llm import Message

            prompt = self._build_prompt(course_text, subject, topic, grade)

            def _call_llm():
                return self.llm.chat([
                    Message(
                        role="system",
                        content=(
                            "你是一位有 20 年教学经验的学科教研员，"
                            "擅长从导学案文本中读出设计者的教学意图。"
                            "请只输出 JSON，不要有任何额外说明。"
                        ),
                    ),
                    Message(role="user", content=prompt),
                ])

            response = await asyncio.wait_for(
                asyncio.to_thread(_call_llm),
                timeout=self.llm_timeout_sec,
            )
            thoughts = self._parse_json_response(response.content, topic, subject)

            logger.info(
                "teaching_thought_extractor.extract_success dimensions=%d",
                len(thoughts),
            )
            return thoughts

        except Exception:
            logger.exception(
                "teaching_thought_extractor.extract_failed timeout_sec=%s",
                self.llm_timeout_sec,
            )
            return self._get_default_thoughts(topic, subject)

    # ── Prompt ────────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        course_text: str,
        subject: str,
        topic: str,
        grade: str,
    ) -> str:
        text_preview = course_text[:4000] if len(course_text) > 4000 else course_text
        grade_info = f"年级：{grade}\n" if grade else ""

        return f"""下面是一份{subject}导学案，课题是"{topic}"。
{grade_info}
请从**设计者视角**拆解这份导学案的教学思路，帮助其他老师理解"为什么这样设计"。

要求：
- 每个维度的 content 用一句话概括（20 字以内）
- key_points 给 2-4 条，每条 15-25 字，直接说结论，不要废话
- 完全基于导学案文本推断，不要编造导学案里没有的信息
- 如果某个维度在导学案里信息不足，如实说"导学案未体现"

请严格按以下 JSON 格式输出，不要输出任何其他内容：

{{
  "what_to_teach": {{
    "content": "一句话说清楚这节课的核心知识点",
    "key_points": [
      "核心概念是什么",
      "学生需要具备的前置知识",
      "本节课的学习目标"
    ]
  }},
  "how_to_open": {{
    "content": "一句话说清楚导入环节的设计方式",
    "key_points": [
      "用了什么情境或问题切入",
      "为什么选这个切入点",
      "与学生已有经验的连接点"
    ]
  }},
  "design_logic": {{
    "content": "一句话说清楚各环节的排列逻辑",
    "key_points": [
      "整体主线是什么",
      "为什么按这个顺序推进",
      "知识点之间的递进关系"
    ]
  }},
  "hard_points": {{
    "content": "一句话说清楚学生最容易卡住的地方",
    "key_points": [
      "易错点1及原因",
      "易错点2及原因",
      "导学案如何处理这个难点"
    ]
  }},
  "how_to_close": {{
    "content": "一句话说清楚收束环节的设计意图",
    "key_points": [
      "用什么方式检验理解",
      "练习题的难度梯度设计",
      "课后延伸的方向"
    ]
  }}
}}

---

导学案内容：
{text_preview}
"""

    # ── 解析 ──────────────────────────────────────────────────────────────────

    def _parse_json_response(
        self, content: str, topic: str, subject: str
    ) -> List[TeachingThought]:
        """解析 LLM 返回的 JSON"""
        # 提取 JSON 块（兼容 LLM 在前后加 markdown 代码块的情况）
        json_str = content.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", json_str)
        if match:
            json_str = match.group(1)
        else:
            # 尝试直接找 { ... }
            brace_match = re.search(r"\{[\s\S]+\}", json_str)
            if brace_match:
                json_str = brace_match.group(0)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("teaching_thought_extractor.json_parse_failed")
            return self._get_default_thoughts(topic, subject)

        thoughts: List[TeachingThought] = []
        for dim_key, icon, title in DIMENSIONS:
            dim_data = data.get(dim_key, {})
            content_text = dim_data.get("content", "").strip()
            key_points = [
                p.strip() for p in dim_data.get("key_points", []) if p.strip()
            ]

            if not content_text:
                content_text = f"{title}：导学案未体现"

            thoughts.append(TeachingThought(
                dimension=dim_key,
                icon=icon,
                title=title,
                content=content_text,
                key_points=key_points,
            ))

        return thoughts

    # ── 降级方案 ──────────────────────────────────────────────────────────────

    def _get_default_thoughts(self, topic: str, subject: str) -> List[TeachingThought]:
        logger.info("teaching_thought_extractor.using_default_thoughts")
        defaults = [
            ("what_to_teach",  "📚", "教什么",   f"{topic} 的核心知识点",
             [f"学科：{subject}", f"课题：{topic}", "详细信息需重新分析"]),
            ("how_to_open",    "🚪", "怎么铺垫", "导入方式待分析",
             ["导学案导入环节信息不足"]),
            ("design_logic",   "🧩", "主线逻辑", "设计逻辑待分析",
             ["环节顺序信息不足"]),
            ("hard_points",    "⚠️",  "难在哪",   "难点待分析",
             ["需结合题型分布进一步分析"]),
            ("how_to_close",   "✅", "怎么收束", "收束方式待分析",
             ["练习题信息不足"]),
        ]
        return [
            TeachingThought(dimension=d, icon=i, title=t, content=c, key_points=kp)
            for d, i, t, c, kp in defaults
        ]


class PhysicsTeachingThoughtExtractor(TeachingThoughtExtractor):
    """物理学科专用，在通用 prompt 基础上补充物理特有的分析角度"""

    def _build_prompt(
        self,
        course_text: str,
        subject: str,
        topic: str,
        grade: str,
    ) -> str:
        base = super()._build_prompt(course_text, subject, topic, grade)
        physics_note = """
分析时额外关注物理学科特点：
- hard_points：物理概念往往反直觉，注意前概念干扰（如"力大速度就大"）
- design_logic：是否有"现象→规律→应用"的推进结构
- how_to_open：是否用了实验、生活情境或反直觉问题切入
"""
        return base + physics_note
