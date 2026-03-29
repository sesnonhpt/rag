"""
Comprehensive lesson template that blends strengths from multiple template styles.
"""

from typing import Any, List

from src.libs.llm.base_llm import Message

from . import TemplateConfig


def build_comprehensive_master_prompt(
    topic: str,
    contexts: List[Any],
    config: TemplateConfig,
    **kwargs,
) -> List[Message]:
    context_text = "\n\n".join(
        f"[{i + 1}] {(r.text or '').strip()[:500]}" for i, r in enumerate(contexts)
    )
    retrieved_images = kwargs.get("retrieved_images", []) or []
    image_lines = []
    for index, image in enumerate(retrieved_images[:6], start=1):
        page_text = f"，原始页码第{image.page}页" if getattr(image, "page", None) else ""
        caption_text = f"，图像说明：{image.caption}" if getattr(image, "caption", None) else ""
        image_lines.append(f"- 配图{index}{page_text}{caption_text}")
    image_context = "\n".join(image_lines)

    blocks = []
    if config.include_background:
        blocks.append(
            "模块A 背景与定位：简要交代主题背景、学科定位、现实价值，控制在精炼可教的范围内。"
        )
    if config.include_facts:
        blocks.append(
            "模块B 核心知识：提炼概念、原理、公式/史实/语言点、易错点，强调条理性和可讲授性。"
        )
    if config.include_examples:
        blocks.append(
            "模块C 教学设计：加入典型例题、课堂活动、讨论问题、分层练习或实验/材料任务。"
        )

    blocks_text = "\n".join(blocks)

    system_prompt = (
        f"你是一名经验丰富的一线教师与教研组长，请为主题“{topic}”生成一份高质量综合模板成稿。\n"
        "这个模板需要综合以下几类模板的优势：\n"
        "1. 学科模板的专业性和学科特色\n"
        "2. 导学案模板的任务驱动、结构清晰、便于学生使用\n"
        "3. 课型模板的教学流程意识\n"
        "4. 功能模板中的练习、总结、知识梳理能力\n"
        "5. 交互模板中的提问、讨论、探究设计\n"
        "6. 个性化模板中的分层表达与适配意识\n\n"
        "要求：\n"
        "- 输出要兼顾教师备课与课堂使用，既能讲，也能学，也能练。\n"
        "- 风格不要过于学术，也不要只剩题单，要做到讲练结合。\n"
        "- 以上下文为主要依据，但不要被上下文束缚成摘要式复述；可以结合通用学科知识、课堂经验、典型案例和教学设计方法做合理拓展。\n"
        "- 允许进行教学性发散补充，例如类比、案例、课堂问题设计、常见误区、应用场景，但不要伪造具体论文出处、教材页码或虚假数据来源。\n"
        "- 自动判断所属学科，并在开头明确写出。\n\n"
        "- 最终成稿必须具备图文并茂的讲解意识，正文中要主动引用“配图1、配图2……”解释图中的关键信息，不能写成纯文字版教案。\n"
        "请严格按以下结构输出：\n"
        "# 《{topic}》综合教学模板\n"
        "学科：<自动判断>\n"
        "适用建议：<建议年级/课型>\n\n"
        "## 一、教学目标\n"
        "- 3条以内，体现知识、能力、思维或素养目标。\n\n"
        "## 二、重点难点\n"
        "- 分别概括重点与难点。\n\n"
        "## 三、导入与背景\n"
        "- 用精炼方式导入主题，可含生活情境、问题情境或知识链接。\n"
        "- 若存在配图，请说明可先展示哪一张图来导入。\n\n"
        "## 四、核心知识梳理\n"
        "- 分点归纳核心概念、规律、史实、方法或语言点。\n"
        "- 每点尽量附一句“易错提醒”或“理解提示”。\n"
        "- 若有配图，至少有2处明确写出“结合配图X观察/讲解……”。\n\n"
        "## 五、典型教学活动/例题\n"
        "- 设计2-4个最有代表性的活动、例题、实验、文本分析或材料任务。\n"
        "- 要体现课堂可执行性。\n\n"
        "## 六、学生任务单\n"
        "- 设计基础任务、提升任务各若干，写成学生可直接完成的指令或题目。\n\n"
        "## 七、课堂互动设计\n"
        "- 提供讨论题、追问问题、合作学习建议或展示任务。\n\n"
        "## 八、分层巩固\n"
        "- 设计基础题、提高题，可标注▲/★。\n\n"
        "## 九、课堂小结\n"
        "- 用2-4条总结本课最重要的知识、方法或思维收获。\n\n"
        "## 十、使用建议\n"
        "- 简要说明教师如何使用这份模板，例如适合新授/复习、哪些部分可删减、哪些适合投屏或纸发。\n\n"
        "写作规则：\n"
        "- 不要输出空泛套话。\n"
        "- 不要写成纯提示词说明。\n"
        "- 内容要完整成稿，适合直接复制使用。\n"
    )

    if blocks_text:
        system_prompt += f"\n内容侧重点：\n{blocks_text}\n"
    if image_context:
        system_prompt += f"\n当前检索到的配图素材：\n{image_context}\n请把这些配图自然地整合进讲解过程。\n"
    else:
        system_prompt += "\n当前没有检索到可用配图，若无图可用再输出纯文字结构。\n"

    return [
        Message(role="system", content=system_prompt.format(topic=topic)),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n请生成主题“{topic}”的综合教学模板。",
        ),
    ]
