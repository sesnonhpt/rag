"""PPT-oriented lesson template."""

from typing import Any, List

from src.libs.llm.base_llm import Message

from . import TemplateConfig


def _build_image_context(retrieved_images: List[Any]) -> str:
    image_lines = []
    for index, image in enumerate((retrieved_images or [])[:6], start=1):
        page_text = f"，原始页码第{image.page}页" if getattr(image, "page", None) else ""
        caption_text = f"，图像说明：{image.caption}" if getattr(image, "caption", None) else ""
        image_lines.append(f"- 配图{index}{page_text}{caption_text}")
    return "\n".join(image_lines)


def build_ppt_master_prompt(
    topic: str,
    contexts: List[Any],
    config: TemplateConfig,
    **kwargs,
) -> List[Message]:
    context_text = "\n\n".join(
        f"[{i + 1}] {(r.text or '').strip()[:500]}" for i, r in enumerate(contexts)
    )
    image_context = _build_image_context(kwargs.get("retrieved_images", []) or [])

    optional_blocks: List[str] = []
    if config.include_background:
        optional_blocks.append("- 前 1-2 页可简要交代背景、价值或生活情境，但不要写成长篇背景介绍。")
    if config.include_facts:
        optional_blocks.append("- 每页优先提炼核心概念、关键结论、规律、公式或易错点。")
    if config.include_examples:
        optional_blocks.append("- 至少安排 2 页案例/活动/练习页，便于课堂讲解和投屏。")
    optional_text = "\n".join(optional_blocks)

    system_prompt = (
        f"你是一名经验丰富的一线教师与教研组长，请为主题“{topic}”生成一份适合直接转成课堂 PPT 的课件成稿。\n"
        "目标不是写长篇教案，而是输出结构化的幻灯片内容。\n"
        "请自动判断学科与适用学段，以上下文为主要依据，并结合通用学科知识做合理补充。\n\n"
        "输出格式必须严格遵守以下规则：\n"
        "1. 全文使用 Markdown。\n"
        "2. 第一行必须是一级标题：# 《{topic}》PPT课件\n"
        "3. 从第二页开始，每一页必须使用二级标题分隔，格式统一为：## 幻灯片N｜页标题\n"
        "4. 每页优先输出 3-5 条要点，使用无序列表 `- `。\n"
        "5. 如某页需要教师讲解提示，可在页末增加一行：讲解提示：……\n"
        "6. 如适合插图，可自然写出“结合配图1观察……”这类表述，但不要伪造图片路径。\n"
        "7. 不要输出 JSON，不要解释模板，不要写成长篇散文。\n\n"
        "整套课件建议 8-12 页，适合投屏展示。\n"
        "常见页面可包含：封面、学习目标、情境导入、核心概念、规律讲解、案例分析、课堂活动、巩固练习、课堂小结。\n"
        "每页内容要短、清楚、可讲，不要把整篇教案一次性塞进一页。\n"
        "如果提供了配图素材，至少安排 1-2 页明确适合图文双栏展示的页面，并在讲解提示中说明图片用途。\n"
    )

    if optional_text:
        system_prompt += f"\n内容控制：\n{optional_text}\n"
    if image_context:
        system_prompt += f"\n当前检索到的配图素材：\n{image_context}\n请优先把这些素材安排到合适的图文页。\n"
    else:
        system_prompt += "\n当前没有检索到可用配图，也要保持完整的课件页结构。\n"

    return [
        Message(role="system", content=system_prompt.format(topic=topic)),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n请生成主题“{topic}”的 PPT 课件成稿。",
        ),
    ]
