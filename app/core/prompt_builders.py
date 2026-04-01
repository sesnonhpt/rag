"""Prompt builder helpers for chat and lesson generation."""

from __future__ import annotations

from typing import Any, List

from src.libs.llm.base_llm import Message


def build_chat_prompt(question: str, contexts: List[Any]) -> List[Message]:
    context_text = "\n\n".join(
        f"[{i+1}] {(r.text or '').strip()[:500]}"
        for i, r in enumerate(contexts)
    )
    return [
        Message(
            role="system",
            content=(
                "你是一个知识库问答助手。请严格基于以下提供的上下文内容回答用户问题。"
                "如果上下文中没有足够信息，请明确告知用户，不要凭空捏造。"
            ),
        ),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n问题：{question}",
        ),
    ]


def build_lesson_plan_prompt(
    topic: str,
    contexts: List[Any],
    include_background: bool,
    include_facts: bool,
    include_examples: bool,
) -> List[Message]:
    context_text = "\n\n".join(
        f"[{i+1}] {(r.text or '').strip()[:500]}"
        for i, r in enumerate(contexts)
    )

    instructions = []
    if include_background:
        instructions.append("""1. 背景信息（详细展开）：
   - 历史背景：详细介绍该主题的历史发展脉络，包括重要时间节点、关键人物和事件
   - 科学意义：阐述该主题在科学史上的重要地位和影响
   - 现实应用：说明该主题在现代社会的应用价值和实际意义
   - 相关概念：介绍与该主题相关的其他重要概念和理论

   **人物背景（如适用）**：
   - 生平简介：详细介绍相关科学家的生平、出生地、教育背景、主要成就
   - 时代背景：介绍科学家所处的时代背景、社会环境、科技发展水平
   - 研究历程：详细描述科学家的研究过程、遇到的困难、突破的关键时刻
   - 人物性格：介绍科学家的性格特点、研究风格、轶事趣闻
   - 历史评价：介绍该科学家在历史上的地位和影响

   **故事背景（如适用）**：
   - 发现过程：详细描述重要发现的过程，包括时间、地点、关键事件
   - 前因后果：介绍发现的背景、动机、以及后续影响
   - 争议与挑战：介绍发现过程中遇到的质疑、争议和挑战
   - 社会反响：介绍发现当时社会的反应和评价
   - 历史意义：阐述该发现对人类文明的深远影响""")

    if include_facts:
        instructions.append("""2. 核心知识点（详细列出）：
   - 基本概念：清晰定义该主题的核心概念和术语
   - 基本原理：详细阐述该主题的基本原理和规律
   - 重要公式：列出相关的数学公式和表达式（如有）
   - 实验现象：描述相关的实验现象和观测结果
   - 常见误区：指出学生容易产生的误解和错误认识

   **相关人物与事件**：
   - 同时代科学家：介绍同时期的其他重要科学家及其贡献
   - 前驱工作：介绍该发现之前的相关研究和理论
   - 后续发展：介绍该发现之后的重要进展和突破
   - 跨学科影响：介绍该发现对其他学科领域的影响""")

    if include_examples:
        instructions.append("""3. 教学活动设计（具体可行）：
   - 演示实验：提供具体的实验设计，包括所需材料、操作步骤和预期结果
   - 课堂活动：设计互动性强的课堂活动，激发学生兴趣
   - 案例分析：提供真实的案例和应用场景
   - 思考问题：设计启发性的思考问题，引导学生深入思考
   - 拓展阅读：推荐相关的课外阅读材料和资源

   **情境教学设计**：
   - 历史重现：设计让学生体验科学家发现过程的活动
   - 角色扮演：设计让学生扮演科学家、进行辩论或讨论的活动
   - 时间线构建：让学生构建相关发现的时间线
   - 对比分析：让学生对比不同科学家的贡献或不同理论的异同
   - 实地考察：推荐相关的博物馆、实验室等实地考察资源""")

    instructions_text = "\n\n".join(instructions)

    return [
        Message(
            role="system",
            content=(
                f"你是一名经验丰富的教师，擅长准备详细、深入的教案。请基于以下上下文内容，为主题'{topic}'生成一份详细的教案。\n"
                "请以上下文内容为主要依据，同时允许结合通用学科知识、课堂经验和教学设计方法做合理拓展。\n"
                "不要伪造具体出处、页码或事实来源；如果上下文不足，可补充教学性内容，但请避免编造可核验细节。\n"
                "最终输出必须为简体中文，不要包含英文句子或英文结尾。\n"
                "首先，请根据主题内容推断这属于哪个学科（如物理、化学、生物、数学、语文、英语、历史、地理等），并在教案开头明确指出。\n\n"
                "请按照以下结构组织教案：\n\n"
                f"{instructions_text}\n\n"
                "4. 参考资源：列出使用的参考资料，包括来源文档和页码"
            ),
        ),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n请为'{topic}'生成一份详细的教案。",
        ),
    ]


def build_lesson_plan_prompt_fallback(req: Any) -> List[Message]:
    instructions = []
    if req.include_background:
        instructions.append("""1. 背景信息（详细展开）：
   - 历史背景：详细介绍该主题的历史发展脉络，包括重要时间节点、关键人物和事件
   - 科学意义：阐述该主题在科学史上的重要地位和影响
   - 现实应用：说明该主题在现代社会的应用价值和实际意义
   - 相关概念：介绍与该主题相关的其他重要概念和理论""")

    if req.include_facts:
        instructions.append("""2. 核心知识点（详细列出）：
   - 基本概念：清晰定义该主题的核心概念和术语
   - 基本原理：详细阐述该主题的基本原理和规律
   - 重要公式：列出相关的数学公式和表达式（如有）
   - 实验现象：描述相关的实验现象和观测结果
   - 常见误区：指出学生容易产生的误解和错误认识""")

    if req.include_examples:
        instructions.append("""3. 教学活动设计（具体可行）：
   - 演示实验：提供具体的实验设计，包括所需材料、操作步骤和预期结果
   - 课堂活动：设计互动性强的课堂活动，激发学生兴趣
   - 案例分析：提供真实的案例和应用场景
   - 思考问题：设计启发性的思考问题，引导学生深入思考
   - 拓展阅读：推荐相关的课外阅读材料和资源""")

    instructions_text = "\n\n".join(instructions)

    return [
        Message(
            role="system",
            content=(
                f"你是一名经验丰富的教师，擅长准备详细、深入的教案。当前知识库中没有找到与主题'{req.topic}'相关的内容，\n"
                "请基于你自己的知识与教学经验为该主题生成一份详细的教案，并主动补充更有教学价值的延伸内容。\n\n"
                "最终输出必须为简体中文，不要包含英文句子或英文结尾。\n"
                "首先，请根据主题内容推断这属于哪个学科（如物理、化学、生物、数学、语文、英语、历史、地理等），并在教案开头明确指出。\n\n"
                "请按照以下结构组织教案：\n\n"
                f"{instructions_text}\n\n"
                "4. 参考资源：列出使用的参考资料（基于你的知识）"
            ),
        ),
        Message(
            role="user",
            content=f"请为'{req.topic}'生成一份详细的教案。",
        ),
    ]
