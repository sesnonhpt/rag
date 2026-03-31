"""Lesson generation agent orchestration."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, List, Optional

from src.core.templates import GradeLevel, LearningStyle, TemplateConfig, TemplateType
from src.libs.llm.base_llm import Message

from .models import ConversationState, LessonAgentAssets, LessonAgentState, LessonReviewReport, QueryPlan
from .planning_models import ExecutionPlan


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
        query_plan: Optional[QueryPlan] = None,
        execution_plan: Optional[ExecutionPlan] = None,
        conversation_state: Optional[ConversationState] = None,
    ) -> LessonAgentState:
        state = LessonAgentState(
            topic=topic,
            template_category=self.request.template_category,
            template_type=self.resolved_template_type,
            query_plan=query_plan,
            execution_plan=execution_plan,
            conversation_state=conversation_state,
        )

        self._retrieve_assets(state, results, image_resources, citations)
        self._generate_draft(state)
        # Keep the review pass enabled by default for better teaching quality.
        # It can still be disabled at runtime with LESSON_REVIEW_ENABLED=false.
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

        if self._looks_like_context_refusal(state.draft_content):
            fallback_messages = self._build_autonomous_planning_messages(state)
            fallback_response = self.llm.chat(fallback_messages)
            state.draft_content = fallback_response.content
            state.metadata["forced_autonomous_retry_after_refusal"] = True

        self.trace.record_stage(
            "agent_generate_draft",
            {
                "template_type": self.resolved_template_type,
                "has_images": len(state.assets.image_resources) > 0,
                "forced_autonomous_retry_after_refusal": bool(state.metadata.get("forced_autonomous_retry_after_refusal")),
            },
            elapsed_ms=elapsed_ms,
        )

    def _insert_images(self, state: LessonAgentState) -> None:
        content = state.final_content or state.draft_content or ""
        usable_context = state.metadata.get("usable_context", True)
        effective_images = state.assets.image_resources if usable_context else []
        if self.resolved_template_type in {"comprehensive_master", "teaching_design_master"} and effective_images:
            content = self.integrate_images(content, effective_images)

        state.final_content = content
        self.trace.record_stage(
            "agent_integrate_images",
            {
                "inserted_image_candidates": len(effective_images),
                "applied": self.resolved_template_type in {"comprehensive_master", "teaching_design_master"} and bool(effective_images),
            },
        )

    def _review_and_polish(self, state: LessonAgentState) -> None:
        draft = state.final_content or state.draft_content or ""
        if not draft:
            return
        review_enabled = str(os.environ.get("LESSON_REVIEW_ENABLED", "true")).strip().lower() not in {"0", "false", "no"}
        if not review_enabled:
            state.final_content = self._clean_trailing_english(draft)
            self.trace.record_stage(
                "agent_review_and_polish",
                {"skipped": True, "reason": "LESSON_REVIEW_ENABLED=false"},
                elapsed_ms=0.0,
            )
            return

        image_count = len(state.assets.image_resources)
        category = state.template_category or "lesson"
        review_mode = self._resolve_review_mode()

        if review_mode == "light":
            messages = [
                Message(
                    role="system",
                    content=(
                        "你是一名资深教研员，请把下面的教案初稿快速润色成更像真实教师成稿的版本。"
                        "要求：保留原有结构，不要大改篇幅；增强课堂推进感、任务感和讲解感；"
                        "如果是教学设计或综合模版(增强版)且已有配图，请自然保留配图讲解位；"
                        "只输出最终成稿，全文必须为简体中文。"
                    ),
                ),
                Message(role="user", content=draft),
            ]
            start_time = time.time()
            response = self.llm.chat(messages)
            elapsed_ms = (time.time() - start_time) * 1000
            state.final_content = self._clean_trailing_english(response.content)
            state.review_notes = []
            self.trace.record_stage(
                "agent_review_and_polish",
                {
                    "template_category": category,
                    "image_count": image_count,
                    "mode": "light",
                },
                elapsed_ms=elapsed_ms,
            )
            return

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
            "4. 如果模板类别是“教学设计”或“综合模版(增强版)”，要确保内容像教案而不是知识点总结，并尽量结合“配图1/配图2”等进行讲解。\n"
            "5. 如果模板类别是“导学案模板”，要增强任务驱动、题目驱动和学生可完成性。\n"
            "6. 如果模板类别是“综合模版(增强版)”，要同时保留教师组织感、学生任务感、互动设计和分层训练。\n"
            "7. 不要输出审稿意见，不要解释修改原因，只输出修订后的最终成稿。\n"
            "8. 全文必须使用简体中文，不要出现英文句子、英文总结语或 'Good luck' 这类英文结尾。\n"
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
        state.final_content = self._clean_trailing_english(response.content)

        if self._looks_like_context_refusal(state.final_content):
            retry_messages = self._build_autonomous_planning_messages(state)
            retry_response = self.llm.chat(retry_messages)
            state.final_content = self._clean_trailing_english(retry_response.content)
            state.metadata["forced_autonomous_retry_after_review"] = True

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
                "mode": "full",
                "must_fix_count": len(review_report.must_fix),
                "forced_autonomous_retry_after_review": bool(state.metadata.get("forced_autonomous_retry_after_review")),
            },
            elapsed_ms=elapsed_ms,
        )

    def _resolve_review_mode(self) -> str:
        configured = str(os.environ.get("LESSON_REVIEW_MODE", "auto")).strip().lower()
        if configured in {"off", "light", "full"}:
            return configured

        model_name = str(getattr(self.request, "model", "") or "").lower()
        if any(token in model_name for token in ["flash-lite", "flash", "mini"]):
            return "light"
        return "full"

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
            "4. 如果是教学设计或综合模版(增强版)，是否真正体现图文并茂；如果已有配图却没有结合配图讲解，要严厉扣分。\n"
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
        max_context_results = int(os.environ.get("LESSON_WRITER_MAX_CONTEXT_RESULTS", "8"))
        forced_autonomous = (
            state.execution_plan is not None
            and str(state.execution_plan.generation_mode or "").strip().lower() == "autonomous"
        )
        has_usable_context = self._has_usable_context_for_topic(self.request.topic, results)
        if forced_autonomous:
            has_usable_context = False
        effective_results = results[: max(1, max_context_results)] if has_usable_context else []
        effective_images = state.assets.image_resources if has_usable_context else []

        self.trace.record_stage(
            "agent_context_relevance_check",
            {
                "raw_result_count": len(results),
                "usable_context": has_usable_context,
                "forced_autonomous": forced_autonomous,
                "effective_result_count": len(effective_results),
                "max_context_results": max(1, max_context_results),
                "effective_image_count": len(effective_images),
            },
        )
        state.metadata["usable_context"] = has_usable_context

        if not has_usable_context:
            return self._build_autonomous_planning_messages(state)

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

                messages = self.template_manager.build_prompt(
                    config=config,
                    topic=self.request.topic,
                    contexts=effective_results,
                    retrieved_images=effective_images,
                )
                if not has_usable_context and messages:
                    messages[0].content += (
                        "\n当前检索到的上下文与主题相关性不足，请不要因为上下文不匹配而拒绝生成。"
                        "请改为基于主题本身进行自主教学规划，结合通用学科知识、课堂经验和教学设计方法完成成稿。"
                    )
                return messages

        if effective_results:
            messages = self.build_default_prompt(
                topic=self.request.topic,
                contexts=effective_results,
                include_background=self.request.include_background,
                include_facts=self.request.include_facts,
                include_examples=self.request.include_examples,
            )
            if not has_usable_context and messages:
                messages[0].content += (
                    "\n若当前上下文与主题不匹配，请忽略这些上下文，直接基于主题自主规划教案，不要输出拒绝语。"
                )
            return messages

        return self.build_fallback_prompt(self.request)

    def _build_autonomous_planning_messages(self, state: LessonAgentState) -> List[Message]:
        template_label = (
            "综合模版(增强版)"
            if self.request.template_category == "comprehensive"
            else ("教学设计" if self.request.template_category == "teaching_design" else "导学案模板")
        )
        system_prompt = (
            f"你是一名经验丰富的一线教师与教研组长。当前知识库未能提供与主题“{self.request.topic}”直接相关的有效上下文，"
            "请不要输出拒绝、说明资料不足、要求用户补充资料等内容。\n"
            f"请直接基于主题本身、通用学科知识、课堂经验和教学设计方法，自主规划并生成一份可直接使用的{template_label}成稿。\n"
            "要求：\n"
            "1. 自动判断所属学科与适用学段。\n"
            "2. 内容必须完整成稿，不能输出解释、免责声明或求补充资料的话术。\n"
            "3. 允许合理发散补充典型案例、类比、实验、活动、误区提醒和应用场景。\n"
            "4. 不要伪造具体教材页码、论文来源或虚假引用。\n"
        )
        if self.request.template_category == "guide":
            system_prompt += (
                "5. 产出风格必须像学校导学案，突出学习目标、重难点、基础部分、要点部分、拓展部分、目标检测，"
                "以学生任务和题目驱动为主。\n"
            )
        elif self.request.template_category == "teaching_design":
            system_prompt += (
                "5. 产出风格必须像学校教学设计，突出教学目标、教学重难点、教学准备、分环节教学过程、板书设计，必要时可补充教学反思。\n"
                "6. 若当前没有可用配图，不要提到配图编号，也不要因为缺图而拒绝生成。\n"
            )
        else:
            system_prompt += (
                "5. 产出风格必须像综合模版(增强版)，既有教学目标、教学过程、板书设计，也有学生任务、互动设计、分层巩固与检测。\n"
                "6. 若当前没有可用配图，不要提到配图编号，也不要因为缺图而拒绝生成。\n"
            )

        return [
            Message(role="system", content=system_prompt),
            Message(role="user", content=f"请直接为主题“{self.request.topic}”生成完整成稿。"),
        ]

    @staticmethod
    def _clean_trailing_english(content: str) -> str:
        """Trim common English tail lines that occasionally leak into Chinese lesson output."""
        text = str(content or "").strip()
        if not text:
            return text

        lines = text.splitlines()
        while lines:
            tail = lines[-1].strip()
            if not tail:
                lines.pop()
                continue
            has_ascii_alpha = bool(re.search(r"[A-Za-z]", tail))
            has_cjk = bool(re.search(r"[\u4e00-\u9fff]", tail))
            is_english_tail = (
                has_ascii_alpha
                and not has_cjk
                and (
                    re.search(r"(good\s+luck|remember\s+to|revised\s+lesson\s+plan)", tail, flags=re.IGNORECASE)
                    or len(re.findall(r"[A-Za-z]", tail)) >= 8
                )
            )
            if is_english_tail:
                lines.pop()
                continue
            break

        return "\n".join(lines).strip()

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

    @staticmethod
    def _has_usable_context_for_topic(topic: str, results: List[Any]) -> bool:
        if not results:
            return False

        topic_terms = LessonAgent._extract_topic_terms(topic)
        if not topic_terms:
            return True

        combined_text = " ".join(str(item.text or "") for item in results[:6]).lower()
        matched_terms = [term for term in topic_terms if term and term in combined_text]
        required_matches = 1 if len(topic_terms) <= 2 else 2
        if len(matched_terms) >= required_matches:
            return True

        return False

    @staticmethod
    def _extract_topic_terms(topic: str) -> List[str]:
        raw_terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]{2,}", str(topic or ""))
        stop_terms = {
            "教案", "模板", "综合模板", "导学案", "综合教学模板", "教学设计", "教学", "内容", "主题",
            "lesson", "guide", "template", "teaching",
        }
        collected: List[str] = []
        for term in raw_terms:
            normalized = term.strip().lower()
            if not normalized or normalized in stop_terms:
                continue
            collected.append(normalized)
            if re.fullmatch(r"[\u4e00-\u9fff]{4,}", normalized):
                length = len(normalized)
                for window in (2, 3, 4):
                    if length <= window:
                        continue
                    for start in range(0, length - window + 1):
                        piece = normalized[start:start + window]
                        if piece not in stop_terms:
                            collected.append(piece)
        return list(dict.fromkeys(collected))

    @staticmethod
    def _looks_like_context_refusal(content: str) -> bool:
        text = str(content or "")
        refusal_patterns = [
            "基于当前上下文，无法",
            "基于提供的上下文，无法",
            "上下文内容并未涉及",
            "与.*没有任何关联",
            "因此，我无法",
            "请补充相关",
            "知识库中没有",
            "无法为",
            "很抱歉",
            "谢谢您的理解",
            "如需该主题的教案",
            "允许使用通用学科知识",
        ]
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in refusal_patterns)
