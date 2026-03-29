"""Lesson generation agent orchestration."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, List, Optional

from src.core.templates import GradeLevel, LearningStyle, TemplateConfig, TemplateType
from src.libs.llm.base_llm import Message

from .models import LessonAgentAssets, LessonAgentState, LessonReviewReport


BuildPromptFn = Callable[[str, List[Any], bool, bool, bool], List[Message]]
BuildFallbackFn = Callable[[Any], List[Message]]
IntegrateImagesFn = Callable[[str, List[Any]], str]


class LessonAgent:
    """Single-agent workflow for lesson generation.

    Flow:
    1. Retrieve assets (already provided by caller)
    2. Generate lesson draft
    3. Review and polish toward a more realistic teaching artifact
    4. Integrate images into the final draft
    """

    def __init__(
        self,
        llm: Any,
        template_manager: Any,
        trace: Any,
        request: Any,
        resolved_template_type: Optional[str],
        build_default_prompt: BuildPromptFn,
        build_fallback_prompt: BuildFallbackFn,
        integrate_images: IntegrateImagesFn,
    ) -> None:
        self.llm = llm
        self.template_manager = template_manager
        self.trace = trace
        self.request = request
        self.resolved_template_type = resolved_template_type
        self.build_default_prompt = build_default_prompt
        self.build_fallback_prompt = build_fallback_prompt
        self.integrate_images = integrate_images

    def run(
        self,
        topic: str,
        results: List[Any],
        image_resources: List[Any],
        citations: List[Any],
    ) -> LessonAgentState:
        state = LessonAgentState(
            topic=topic,
            template_category=self.request.template_category,
            template_type=self.resolved_template_type,
        )

        self._retrieve_assets(state, results, image_resources, citations)
        self._generate_draft(state)
        self._review_and_polish(state)
        self._insert_images(state)
        self._extract_subject(state)
        return state

    def _retrieve_assets(
        self,
        state: LessonAgentState,
        results: List[Any],
        image_resources: List[Any],
        citations: List[Any],
    ) -> None:
        state.assets = LessonAgentAssets(
            text_results=results,
            image_resources=image_resources,
            citations=citations,
        )
        self.trace.record_stage(
            "agent_retrieve_assets",
            {
                "text_result_count": len(results),
                "image_count": len(image_resources),
                "citation_count": len(citations),
            },
        )

    def _generate_draft(self, state: LessonAgentState) -> None:
        messages = self._build_generation_messages(state)
        start_time = time.time()
        response = self.llm.chat(messages)
        elapsed_ms = (time.time() - start_time) * 1000
        state.draft_content = response.content

        self.trace.record_stage(
            "agent_generate_draft",
            {
                "template_type": self.resolved_template_type,
                "has_images": len(state.assets.image_resources) > 0,
            },
            elapsed_ms=elapsed_ms,
        )

    def _insert_images(self, state: LessonAgentState) -> None:
        content = state.final_content or state.draft_content or ""
        if self.resolved_template_type == "comprehensive_master" and state.assets.image_resources:
            content = self.integrate_images(content, state.assets.image_resources)

        state.final_content = content
        self.trace.record_stage(
            "agent_integrate_images",
            {
                "inserted_image_candidates": len(state.assets.image_resources),
                "applied": self.resolved_template_type == "comprehensive_master" and bool(state.assets.image_resources),
            },
        )

    def _review_and_polish(self, state: LessonAgentState) -> None:
        draft = state.final_content or state.draft_content or ""
        if not draft:
            return

        image_count = len(state.assets.image_resources)
        category = state.template_category or "lesson"
        review_report = self._assess_draft(draft, category, image_count)
        state.review_report = review_report
        state.review_notes = list(review_report.must_fix or review_report.issues)

        must_fix_lines = "\n".join(f"- {item}" for item in review_report.must_fix) or "- 无强制问题"
        issue_lines = "\n".join(f"- {item}" for item in review_report.issues) or "- 无明显问题"
        score_line = (
            f"真实教案感={review_report.realism_score}/10，"
            f"教学设计={review_report.pedagogy_score}/10，"
            f"结构完整度={review_report.structure_score}/10，"
            f"图文融合={review_report.multimodal_score}/10"
        )

        review_prompt = (
            "你是一名资深教研员，请对下面的教案/导学案初稿进行定向修订，使其更像真实学校教师写出的最终成稿。\n"
            "修订原则：\n"
            "1. 优先解决“必须修复项”，其次解决一般问题。\n"
            "2. 避免资料摘要口吻，增强课堂感和教学组织感。\n"
            "3. 允许在不伪造具体出处的前提下，补充合理的教学性发散内容，例如案例、类比、误区提醒、活动组织语。\n"
            "4. 如果模板类别是“综合模板”，要确保内容像教案而不是知识点总结，并尽量结合“配图1/配图2”等进行讲解。\n"
            "5. 如果模板类别是“导学案模板”，要增强任务驱动、题目驱动和学生可完成性。\n"
            "6. 不要输出审稿意见，不要解释修改原因，只输出修订后的最终成稿。\n"
        )
        review_prompt += (
            f"\n当前模板类别：{category}\n"
            f"当前可用配图数量：{image_count}\n"
            f"当前评估得分：{score_line}\n"
            f"必须修复项：\n{must_fix_lines}\n"
            f"一般问题：\n{issue_lines}\n"
        )

        messages = [
            Message(role="system", content=review_prompt),
            Message(role="user", content=draft),
        ]
        start_time = time.time()
        response = self.llm.chat(messages)
        elapsed_ms = (time.time() - start_time) * 1000
        state.final_content = response.content

        self.trace.record_stage(
            "agent_review_and_polish",
            {
                "template_category": category,
                "image_count": image_count,
                "review_scores": {
                    "realism": review_report.realism_score,
                    "pedagogy": review_report.pedagogy_score,
                    "structure": review_report.structure_score,
                    "multimodal": review_report.multimodal_score,
                },
                "must_fix_count": len(review_report.must_fix),
            },
            elapsed_ms=elapsed_ms,
        )

    def _assess_draft(
        self,
        draft: str,
        category: str,
        image_count: int,
    ) -> LessonReviewReport:
        system_prompt = (
            "你是一名极其严格的教案质检员，请对下面的初稿做结构化评估。\n"
            "请只输出 JSON，不要输出 Markdown，不要解释。\n"
            "JSON 结构必须是：\n"
            "{"
            "\"realism_score\": int,"
            "\"pedagogy_score\": int,"
            "\"structure_score\": int,"
            "\"multimodal_score\": int,"
            "\"strengths\": [str, ...],"
            "\"issues\": [str, ...],"
            "\"must_fix\": [str, ...]"
            "}\n"
            "评分标准：0-10 分。\n"
            "评估重点：\n"
            "1. 是否像真实教师会写的教案/导学案，而不是资料总结。\n"
            "2. 是否有明确教学设计、任务设计、课堂推进感。\n"
            "3. 结构是否完整清楚。\n"
            "4. 如果是综合模板，是否真正体现图文并茂；如果已有配图却没有结合配图讲解，要严厉扣分。\n"
            "5. must_fix 只列最关键、最值得修改的问题，最多 5 条。\n"
        )
        user_prompt = (
            f"模板类别：{category}\n"
            f"当前可用配图数量：{image_count}\n\n"
            f"初稿内容：\n{draft}"
        )
        response = self.llm.chat(
            [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ]
        )
        payload = self._parse_json_object(response.content)
        return LessonReviewReport(
            realism_score=self._safe_int(payload.get("realism_score")),
            pedagogy_score=self._safe_int(payload.get("pedagogy_score")),
            structure_score=self._safe_int(payload.get("structure_score")),
            multimodal_score=self._safe_int(payload.get("multimodal_score")),
            strengths=self._safe_str_list(payload.get("strengths")),
            issues=self._safe_str_list(payload.get("issues")),
            must_fix=self._safe_str_list(payload.get("must_fix")),
        )

    def _extract_subject(self, state: LessonAgentState) -> None:
        content = state.final_content or state.draft_content or ""
        subject = None
        for line in content.splitlines()[:20]:
            if "学科：" in line or "学科:" in line:
                subject = line.split("：")[-1].split(":")[-1].strip()
                break
            if "计算机" in line and ("教案" in line or "模板" in line or "导学案" in line):
                subject = "计算机科学"
                break
            if "物理" in line and ("教案" in line or "模板" in line or "导学案" in line):
                subject = "物理"
                break
            if "化学" in line and ("教案" in line or "模板" in line or "导学案" in line):
                subject = "化学"
                break
            if "生物" in line and ("教案" in line or "模板" in line or "导学案" in line):
                subject = "生物"
                break
            if "数学" in line and ("教案" in line or "模板" in line or "导学案" in line):
                subject = "数学"
                break
            if "语文" in line and ("教案" in line or "模板" in line or "导学案" in line):
                subject = "语文"
                break
            if "英语" in line and ("教案" in line or "模板" in line or "导学案" in line):
                subject = "英语"
                break
            if "历史" in line and ("教案" in line or "模板" in line or "导学案" in line):
                subject = "历史"
                break
            if "地理" in line and ("教案" in line or "模板" in line or "导学案" in line):
                subject = "地理"
                break
        state.subject = subject

    def _build_generation_messages(self, state: LessonAgentState) -> List[Message]:
        results = state.assets.text_results
        if self.resolved_template_type:
            template_type = self._resolve_template_enum(self.resolved_template_type)
            if template_type is not None:
                config = TemplateConfig(
                    template_type=template_type,
                    include_background=self.request.include_background,
                    include_facts=self.request.include_facts,
                    include_examples=self.request.include_examples,
                )

                if getattr(self.request, "grade_level", None):
                    grade_map = {
                        "primary": GradeLevel.PRIMARY,
                        "middle": GradeLevel.MIDDLE,
                        "high": GradeLevel.HIGH,
                        "college": GradeLevel.COLLEGE,
                    }
                    config.grade_level = grade_map.get(self.request.grade_level, GradeLevel.MIDDLE)

                if getattr(self.request, "learning_style", None):
                    style_map = {
                        "visual": LearningStyle.VISUAL,
                        "auditory": LearningStyle.AUDITORY,
                        "kinesthetic": LearningStyle.KINESTHETIC,
                        "read_write": LearningStyle.READ_WRITE,
                    }
                    config.learning_style = style_map.get(self.request.learning_style, LearningStyle.VISUAL)

                return self.template_manager.build_prompt(
                    config=config,
                    topic=self.request.topic,
                    contexts=results if results else [],
                    retrieved_images=state.assets.image_resources,
                )

        if results:
            return self.build_default_prompt(
                topic=self.request.topic,
                contexts=results,
                include_background=self.request.include_background,
                include_facts=self.request.include_facts,
                include_examples=self.request.include_examples,
            )

        return self.build_fallback_prompt(self.request)

    @staticmethod
    def _resolve_template_enum(template_type_str: str) -> Optional[TemplateType]:
        for template_type in TemplateType:
            if template_type.value == template_type_str:
                return template_type
        return None

    @staticmethod
    def _parse_json_object(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except Exception:
            return 0

    @staticmethod
    def _safe_str_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]
