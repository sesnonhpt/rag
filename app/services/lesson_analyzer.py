"""
导学案分析服务

面向旧导学案试点，输出更贴近老师使用场景的教学设计拆解结果。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from src.observability.logger import get_logger

logger = get_logger(__name__)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_list(values: Any, fallback: List[str]) -> List[str]:
    if not isinstance(values, list):
        return list(fallback)
    cleaned = [str(item).strip() for item in values if str(item or "").strip()]
    return cleaned or list(fallback)


def _clean_question_types(value: Any) -> Dict[str, int]:
    data = value if isinstance(value, dict) else {}
    return {
        "choice": int(data.get("choice", 0) or 0),
        "fill": int(data.get("fill", 0) or 0),
        "application": int(data.get("application", 0) or 0),
        "other": int(data.get("other", 0) or 0),
    }


@dataclass
class LessonAnalysis:
    """导学案分析结果。"""

    topic: str
    subject: str
    grade: str
    difficulty: str
    question_types: Dict[str, int]
    teaching_sections: List[str]
    key_points: List[str]
    lesson_mainline: str
    design_logic: List[str]
    teacher_moves: List[str]
    student_difficulties: List[str]
    activity_flow: List[str]
    edit_priorities: List[str]
    reusable_sections: List[str]
    skeleton_markdown: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LessonAnalyzer:
    """导学案分析器。"""

    def __init__(self, llm):
        self.llm = llm
        self.llm_timeout_sec = float(os.environ.get("GUIDE_ANALYZER_LLM_TIMEOUT_SEC", "20"))

    async def analyze(self, content: str) -> LessonAnalysis:
        """分析导学案内容。"""
        logger.info("lesson_analyzer.analyze_start content_length=%d", len(content))

        try:
            from src.libs.llm.base_llm import Message
            import asyncio

            prompt = self._build_prompt(content)

            def _call_llm():
                return self.llm.chat([
                    Message(
                        role="system",
                        content="你是资深教研员，擅长把旧导学案拆成老师能直接接着修改的教学设计信息。",
                    ),
                    Message(role="user", content=prompt),
                ])

            response = await asyncio.wait_for(
                asyncio.to_thread(_call_llm),
                timeout=self.llm_timeout_sec,
            )
            analysis = self._parse_response(response.content, content)

            logger.info(
                "lesson_analyzer.analyze_success topic=%s difficulty=%s",
                analysis.topic,
                analysis.difficulty,
            )
            return analysis

        except Exception:
            logger.exception(
                "lesson_analyzer.analyze_failed timeout_sec=%s",
                self.llm_timeout_sec,
            )
            return self._get_basic_analysis(content)

    def _build_prompt(self, content: str) -> str:
        text_preview = content[:6000] if len(content) > 6000 else content

        return f"""请分析这份旧导学案，并输出一份老师可直接使用的“教学设计拆解结果”。

要求：
1. 只输出 JSON，不要输出其他说明。
2. 重点不是学术分析，而是帮新老师看懂这份导学案背后的设计思路。
3. 如果信息不完整，请结合常见课堂做合理推断，但不要夸张。
4. `skeleton_markdown` 必须是一份可继续编辑的导学案骨架，用 Markdown 输出。
5. 所有列表尽量控制在 3 到 5 条。

输出格式：
{{
  "topic": "知识点名称",
  "subject": "学科",
  "grade": "年级",
  "difficulty": "简单/中等/困难",
  "question_types": {{
    "choice": 0,
    "fill": 0,
    "application": 0,
    "other": 0
  }},
  "teaching_sections": ["导入", "概念建构", "练习巩固"],
  "key_points": ["重点1", "重点2", "重点3"],
  "lesson_mainline": "用 1-2 句话概括这节课真正的课堂主线",
  "design_logic": ["为什么这样导入", "为什么先讲这个再讲那个", "为什么这样收束"],
  "teacher_moves": ["老师可以直接借鉴的话术或动作1", "老师可以直接借鉴的话术或动作2"],
  "student_difficulties": ["学生可能卡住的点1", "学生可能卡住的点2"],
  "activity_flow": ["第1环节怎么做", "第2环节怎么做", "第3环节怎么做"],
  "edit_priorities": ["新老师优先补什么", "哪里最值得改", "哪些内容可直接沿用"],
  "reusable_sections": ["可以直接复用的部分1", "可以直接复用的部分2"],
  "skeleton_markdown": "# 课题\\n## 一、教材与学情\\n..."
}}

旧导学案内容：
{text_preview}
"""

    def _parse_response(self, content: str, original_content: str) -> LessonAnalysis:
        try:
            json_text = content.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            data = json.loads(json_text.strip())
        except Exception as exc:
            logger.warning("lesson_analyzer.parse_failed error=%s", str(exc))
            return self._get_basic_analysis(original_content)

        topic = _clean_text(data.get("topic")) or self._infer_topic(original_content)
        subject = _clean_text(data.get("subject")) or self._infer_subject(original_content)
        grade = _clean_text(data.get("grade")) or "未明确年级"
        key_points = _clean_list(data.get("key_points"), ["待补充本课核心知识点"])
        teaching_sections = _clean_list(
            data.get("teaching_sections"),
            ["导入", "概念讲解", "练习巩固"],
        )
        lesson_mainline = _clean_text(data.get("lesson_mainline")) or "先导入情境，再建立概念，最后通过练习收束。"
        design_logic = _clean_list(
            data.get("design_logic"),
            ["先用学生熟悉的情境引出问题。", "再通过例题或活动把知识点讲透。", "最后用练习检查理解。"],
        )
        teacher_moves = _clean_list(
            data.get("teacher_moves"),
            ["先追问学生已有认识。", "边讲边让学生说出判断依据。"],
        )
        student_difficulties = _clean_list(
            data.get("student_difficulties"),
            ["学生容易只记结论，不理解形成过程。"],
        )
        activity_flow = _clean_list(
            data.get("activity_flow"),
            ["情境导入", "概念建构", "迁移练习"],
        )
        edit_priorities = _clean_list(
            data.get("edit_priorities"),
            ["先补齐课堂主线。", "再把关键提问写得更像课堂语言。", "最后补充练习和板书。"],
        )
        reusable_sections = _clean_list(
            data.get("reusable_sections"),
            ["知识点顺序可直接沿用。"],
        )
        skeleton_markdown = _clean_text(data.get("skeleton_markdown")) or self._build_fallback_skeleton(
            topic=topic,
            subject=subject,
            grade=grade,
            key_points=key_points,
            activity_flow=activity_flow,
            student_difficulties=student_difficulties,
        )

        return LessonAnalysis(
            topic=topic,
            subject=subject,
            grade=grade,
            difficulty=_clean_text(data.get("difficulty")) or "中等",
            question_types=_clean_question_types(data.get("question_types")),
            teaching_sections=teaching_sections,
            key_points=key_points,
            lesson_mainline=lesson_mainline,
            design_logic=design_logic,
            teacher_moves=teacher_moves,
            student_difficulties=student_difficulties,
            activity_flow=activity_flow,
            edit_priorities=edit_priorities,
            reusable_sections=reusable_sections,
            skeleton_markdown=skeleton_markdown,
        )

    def _get_basic_analysis(self, content: str) -> LessonAnalysis:
        logger.info("lesson_analyzer.using_basic_analysis")

        topic = self._infer_topic(content)
        subject = self._infer_subject(content)
        grade = self._infer_grade(content)
        question_types = {
            "choice": min(len(re.findall(r"[A-D][\s\.、]", content)), 20),
            "fill": min(len(re.findall(r"_{3,}|（\s*）", content)), 20),
            "application": min(len(re.findall(r"解答|计算|探究|证明", content)), 20),
            "other": 0,
        }
        key_points = self._infer_key_points(content, subject)
        activity_flow = ["情境导入", "核心知识展开", "当堂练习与总结"]
        student_difficulties = ["学生容易记住结论，但忽略概念形成过程。"]

        return LessonAnalysis(
            topic=topic,
            subject=subject,
            grade=grade,
            difficulty="中等",
            question_types=question_types,
            teaching_sections=["导入", "新知讲解", "练习巩固"],
            key_points=key_points,
            lesson_mainline="导学案整体更像是先回顾基础，再推进新知，最后通过练习完成收束。",
            design_logic=[
                "先用旧知或情境把学生带进课题。",
                "再按知识点顺序逐步展开，避免学生一下子吃太多。",
                "最后通过练习和总结检验学生是否真正理解。",
            ],
            teacher_moves=[
                "先问学生已有经验，再进入新知识。",
                "每讲完一个关键点，就安排一个小练习或追问。",
            ],
            student_difficulties=student_difficulties,
            activity_flow=activity_flow,
            edit_priorities=[
                "先补清楚每个环节为什么这么安排。",
                "把提问改成老师课堂里真的会说的话。",
                "补齐易错提醒和练习收束。",
            ],
            reusable_sections=["知识点顺序", "部分例题素材"],
            skeleton_markdown=self._build_fallback_skeleton(
                topic=topic,
                subject=subject,
                grade=grade,
                key_points=key_points,
                activity_flow=activity_flow,
                student_difficulties=student_difficulties,
            ),
        )

    def _infer_subject(self, content: str) -> str:
        subject_keywords = {
            "物理": ["力", "运动", "能量", "电", "磁", "光", "加速度", "惯性"],
            "化学": ["化学", "反应", "元素", "分子", "原子", "酸", "碱"],
            "生物": ["生物", "细胞", "遗传", "进化", "生态", "植物", "动物"],
            "数学": ["方程", "函数", "几何", "代数", "概率", "统计"],
            "语文": ["课文", "阅读", "写作", "古诗", "文言文"],
            "英语": ["English", "单词", "语法", "阅读理解"],
        }
        for subject, keywords in subject_keywords.items():
            if any(keyword in content for keyword in keywords):
                return subject
        return "未知学科"

    def _infer_grade(self, content: str) -> str:
        match = re.search(r"(小学|初中|高中)([一二三四五六七八九])?", content)
        if match:
            return "".join([part for part in match.groups() if part])
        return "未明确年级"

    def _infer_topic(self, content: str) -> str:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return "未命名课题"
        first_line = lines[0]
        first_line = re.sub(r"^[\d一二三四五六七八九十.\-、\s]+", "", first_line)
        return first_line[:40] or "未命名课题"

    def _infer_key_points(self, content: str, subject: str) -> List[str]:
        fallback_map = {
            "物理": ["概念形成过程", "实验或现象观察", "规律应用与辨析"],
            "数学": ["核心概念理解", "典型题型迁移", "易错点辨析"],
            "化学": ["核心原理理解", "实验现象观察", "条件与结论区分"],
        }
        fallback = fallback_map.get(subject, ["核心知识点梳理", "课堂提问设计", "练习与巩固"])
        return fallback

    def _build_fallback_skeleton(
        self,
        *,
        topic: str,
        subject: str,
        grade: str,
        key_points: List[str],
        activity_flow: List[str],
        student_difficulties: List[str],
    ) -> str:
        key_points_text = "\n".join([f"- {item}" for item in key_points[:4]])
        flow_text = "\n".join([f"1. {item}" for item in activity_flow[:4]])
        difficulty_text = "\n".join([f"- {item}" for item in student_difficulties[:4]])
        return f"""# {topic}

## 一、教材与学情判断
- 学科：{subject}
- 年级：{grade}
- 本课核心：请老师补充本课的真实教学定位。

## 二、教学目标
- 知识与理解：
- 过程与方法：
- 表达与迁移：

## 三、教学重点与难点
### 重点
{key_points_text}

### 难点
{difficulty_text}

## 四、课堂主线
- 请围绕“导入 - 建构 - 巩固 - 收束”补充本课主线。

## 五、课堂流程
{flow_text}

## 六、关键提问与师生活动
- 导入提问：
- 推进提问：
- 收束提问：

## 七、板书与练习建议
- 板书主框架：
- 当堂练习：
- 课后延伸：
"""
