"""
Teaching-design and enhanced comprehensive lesson templates.
"""

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


def build_teaching_design_prompt(
    topic: str,
    contexts: List[Any],
    config: TemplateConfig,
    **kwargs,
) -> List[Message]:
    context_text = "\n\n".join(
        f"[{i + 1}] {(r.text or '').strip()[:500]}" for i, r in enumerate(contexts)
    )

    system_prompt = (
        f"你是一名经验丰富的一线教师与教研组长，请为主题“{topic}”生成一份高质量“教学设计”成稿。\n"
        "要求：\n"
        "- 输出要明显偏教师备课稿，贴近学校常见“教学设计”写法，而不是学生任务单。\n"
        "- 风格不要过于学术，也不要只剩题单，要做到讲练结合、过程清楚、环节完整。\n"
        "- 以上下文为主要依据，但不要被上下文束缚成摘要式复述；可以结合通用学科知识、课堂经验、典型案例和教学设计方法做合理拓展。\n"
        "- 自动判断所属学科与适用学段。\n\n"
        "请严格按以下结构输出：\n"
        "# 《{topic}》教学设计\n"
        "一、教学目标\n"
        "二、教学重难点\n"
        "教学准备：<可按学科补充课件、词卡、实验器材、评价量表等>\n"
        "二、教学过程\n"
        "（一）<导入环节标题>（约X分钟）\n"
        "（二）<新授/探究环节标题>（约X分钟）\n"
        "（三）<训练/感悟/巩固环节标题>（约X分钟）\n"
        "（四）<应用/书写/总结环节标题>（约X分钟）\n"
        "（五）<结课与延伸环节标题>（约X分钟）\n"
        "三、板书设计\n"
        "设计说明：<可选>\n"
        "教学反思：<可选，若合适再写>\n\n"
        "写作规则：\n"
        "- 内容要完整成稿，适合直接复制使用。\n"
        "- “教学过程”必须写出分环节标题、师生活动、关键提问或任务、设计意图。\n"
        "- 每个教学环节尽量包含：活动内容、教师引导、学生任务、预期目标或设计意图。\n"
        "- 风格贴近《动物儿歌》《蜘蛛开店》这类真实样例。\n"
        "- 这是教师使用的教学设计，不要写成学生答题单，也不要使用【基础部分】【目标检测】这类导学案主栏目。\n"
    )

    return [
        Message(role="system", content=system_prompt.format(topic=topic)),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n请生成主题“{topic}”的教学设计。",
        ),
    ]


def build_comprehensive_master_prompt(
    topic: str,
    contexts: List[Any],
    config: TemplateConfig,
    **kwargs,
) -> List[Message]:
    context_text = "\n\n".join(
        f"[{i + 1}] {(r.text or '').strip()[:500]}" for i, r in enumerate(contexts)
    )
    image_context = _build_image_context(kwargs.get("retrieved_images", []) or [])

    blocks = []
    if config.include_background:
        blocks.append("模块A 背景与定位：简要交代主题背景、学科定位、现实价值。")
    if config.include_facts:
        blocks.append("模块B 核心知识：提炼概念、原理、公式/史实/语言点、易错点。")
    if config.include_examples:
        blocks.append("模块C 教学设计：加入典型例题、课堂活动、讨论问题、分层练习或实验/材料任务。")
    blocks_text = "\n".join(blocks)

    system_prompt = (
        f"你是一名经验丰富的一线教师与教研组长，请为主题“{topic}”生成一份“综合模版(增强版)”成稿。\n"
        "这份模板要融合三类长处：\n"
        "1. 教学设计的课堂流程与教师组织语言\n"
        "2. 导学案的任务驱动、题目驱动与学生可完成性\n"
        "3. 增强版模板的分层训练、互动设计、配图讲解与可直接落地使用感\n\n"
        "要求：\n"
        "- 输出是增强版综合成稿，既像正式教案，又保留学生任务与课堂练习位。\n"
        "- 不能写成单纯教学设计，也不能退化成单纯导学案。\n"
        "- 要有更强的课堂推进感、任务层次感、讲练结合感。\n"
        "- 以上下文为主要依据，但允许结合通用学科知识、课堂经验、典型案例和教学设计方法做合理拓展。\n"
        "- 自动判断所属学科与适用学段。\n"
        "- 如存在配图，正文中要自然引用“配图1、配图2……”解释关键信息；如无图，也要保持完整结构。\n\n"
        "请严格按以下结构输出：\n"
        "# 《{topic}》综合模版（增强版）\n"
        "学科：<自动判断>\n"
        "适用建议：<建议年级/课型>\n\n"
        "## 一、教学目标\n"
        "- 3条以内，体现知识、能力、思维或素养目标。\n\n"
        "## 二、教学重难点\n"
        "- 分别概括重点与难点。\n\n"
        "## 三、教学准备\n"
        "- 课件、材料、任务单、实验器材、配图或评价工具。\n\n"
        "## 四、教学过程\n"
        "- 按 4-6 个环节展开，每个环节写清教师活动、学生活动、关键问题、设计意图。\n"
        "- 如有配图，至少有2处明确写出“结合配图X观察/讲解……”。\n\n"
        "## 五、学生任务单\n"
        "- 设计基础任务、提升任务各若干，写成学生可直接完成的指令或题目。\n\n"
        "## 六、课堂互动设计\n"
        "- 提供讨论题、追问问题、合作学习建议或展示任务。\n\n"
        "## 七、分层巩固与检测\n"
        "- 设计基础题、提高题、拓展题，可标注▲/★。\n\n"
        "## 八、课堂小结\n"
        "- 用2-4条总结本课知识、方法或思维收获。\n\n"
        "## 九、板书设计\n"
        "- 概括板书结构与关键词。\n\n"
        "## 十、使用建议\n"
        "- 简要说明教师如何使用这份增强版模板，例如哪些内容适合投屏、哪些适合纸发、哪些环节可删减。\n\n"
        "写作规则：\n"
        "- 不要输出空泛套话。\n"
        "- 不要写成纯提示词说明。\n"
        "- 内容要完整成稿，适合直接复制使用。\n"
        "- 相比普通教学设计，要更强调互动、任务、训练和分层。\n"
        "- 相比导学案，要更强调教师组织、教学推进与课堂讲解。\n"
    )

    if blocks_text:
        system_prompt += f"\n内容侧重点：\n{blocks_text}\n"
    if image_context:
        system_prompt += f"\n当前检索到的配图素材：\n{image_context}\n请把这些配图自然地整合进讲解过程。\n"
    else:
        system_prompt += "\n当前没有检索到可用配图，可输出纯文字增强版结构，但不要缩减栏目。\n"

    return [
        Message(role="system", content=system_prompt.format(topic=topic)),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n请生成主题“{topic}”的综合模版（增强版）。",
        ),
    ]
