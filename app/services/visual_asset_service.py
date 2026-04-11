"""Generated visual asset planning and integration helpers."""

from __future__ import annotations

import json
from typing import Any, List, Optional

from app.schemas.api_models import LessonImageResource
from app.services.image_generation_service import ExperimentalImageGenerationService, ImageGenerationError
from src.libs.llm.base_llm import Message
from src.observability.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED_TEMPLATE_CATEGORIES = {"comprehensive", "teaching_design"}
_AUTO_VISUAL_POSITIVE_KEYWORDS = (
    "多图片",
    "更多图片",
    "多一点图片",
    "增加图片",
    "增加配图",
    "增加插图",
    "图文并茂",
    "图文结合",
    "需要图片",
    "需要配图",
    "需要插图",
    "配图",
    "插图",
    "示意图",
    "流程图",
    "结构图",
    "图解",
    "补图",
    "可视化",
)
_AUTO_VISUAL_NEGATIVE_KEYWORDS = (
    "不要图片",
    "不需要图片",
    "不要配图",
    "不需要配图",
    "不要插图",
    "不需要插图",
    "纯文字",
    "无图",
    "不要ai图",
    "不要ai生图",
    "不要ai示意图",
    "不需要ai图",
    "不需要ai生图",
    "不需要ai示意图",
    "禁用ai图",
)


def should_auto_generate_visuals_from_notes(notes: Optional[str]) -> bool:
    normalized_notes = "".join(str(notes or "").lower().split())
    if not normalized_notes:
        return False
    if any(keyword in normalized_notes for keyword in _AUTO_VISUAL_NEGATIVE_KEYWORDS):
        return False
    return any(keyword in normalized_notes for keyword in _AUTO_VISUAL_POSITIVE_KEYWORDS)


def detect_visual_generation_intent(notes: Optional[str], llm: Any | None = None) -> bool:
    normalized_notes = " ".join(str(notes or "").split())
    if not normalized_notes:
        return False

    heuristic_decision = should_auto_generate_visuals_from_notes(normalized_notes)
    if llm is None:
        return heuristic_decision

    messages = [
        Message(
            role="system",
            content=(
                "你是教案生成系统里的备注意图分类器。"
                "请判断用户备注是否表达了“希望补充 AI 教学示意图/流程图/结构图/插图”等可视化增强意图。"
                "只输出 JSON，不要解释。"
                '格式必须为 {"generate_visual": true|false, "reason": "简短原因"}。'
                "如果用户明确要求不要图片、不要AI图、纯文字，则返回 false。"
                "如果用户只是泛泛要求讲清楚重点、增加篇幅、强化内容，但没有体现想要图示增强，返回 false。"
            ),
        ),
        Message(
            role="user",
            content=f"备注：{normalized_notes}",
        ),
    ]
    try:
        response = llm.chat(messages).content.strip()
        payload = json.loads(response)
        decision = bool(payload.get("generate_visual"))
        logger.info(
            "lesson_visual.intent_detected notes=%s decision=%s reason=%s mode=llm",
            normalized_notes,
            decision,
            str(payload.get("reason") or ""),
        )
        return decision
    except Exception as exc:
        logger.warning(
            "lesson_visual.intent_detect_failed notes=%s fallback=%s error=%s",
            normalized_notes,
            heuristic_decision,
            exc,
        )
        return heuristic_decision


def should_generate_visual_asset(
    *,
    topic: str,
    notes: Optional[str],
    template_category: Optional[str],
    existing_images: List[LessonImageResource],
    llm: Any | None = None,
) -> bool:
    _ = existing_images
    if not detect_visual_generation_intent(notes, llm=llm):
        return False
    if not str(topic or "").strip():
        return False
    if str(template_category or "").strip() not in _SUPPORTED_TEMPLATE_CATEGORIES:
        return False
    return True


def build_visual_generation_prompt(
    *,
    topic: str,
    notes: Optional[str],
    template_category: Optional[str],
) -> tuple[str, str]:
    style = "diagram_clean"
    notes_text = " ".join(str(notes or "").split())
    category_text = str(template_category or "comprehensive").strip()
    topic_text = " ".join(str(topic or "").split())

    if any(keyword in notes_text for keyword in ("导入", "情境", "课堂", "插画")):
        style = "education_illustration"

    prompt = (
        f"为“{topic_text}”生成一张适合{category_text}教案使用的中文教学示意图。"
        "要求体现核心概念关系或关键过程，画面结构清楚，中文标签简洁，适合直接插入教案或 DOCX。"
    )
    if notes_text:
        prompt += f" 额外要求：{notes_text}"
    return prompt, style


def maybe_generate_visual_asset(
    *,
    topic: str,
    notes: Optional[str],
    template_category: Optional[str],
    existing_images: List[LessonImageResource],
    llm: Any | None = None,
) -> Optional[LessonImageResource]:
    if not should_generate_visual_asset(
        topic=topic,
        notes=notes,
        template_category=template_category,
        existing_images=existing_images,
        llm=llm,
    ):
        return None

    prompt, style = build_visual_generation_prompt(
        topic=topic,
        notes=notes,
        template_category=template_category,
    )
    service = ExperimentalImageGenerationService()
    try:
        result = service.generate_image(
            prompt=prompt,
            style=style,
            topic=topic,
        )
    except ImageGenerationError as exc:
        logger.warning(
            "lesson_visual.generated_failed topic=%s template_category=%s error=%s",
            topic,
            template_category or "",
            exc,
        )
        return None

    image_id = str(result["filename"]).rsplit(".", 1)[0]
    return LessonImageResource(
        image_id=image_id,
        url=result["image_url"],
        source="AI 示意图",
        page=None,
        caption=f"AI 生成教学示意图：{topic}",
        source_type="generated",
        role="supporting_visual",
        model=result.get("model"),
    )
