"""
Q&A Templates for Modular RAG MCP Server.

This module provides various Q&A prompt templates for different answer styles.
"""

from typing import Any, List
from src.libs.llm.base_llm import Message
from . import TemplateConfig


def build_structured_qa_prompt(
    topic: str,
    contexts: List[Any],
    config: TemplateConfig,
    **kwargs
) -> List[Message]:
    """Build structured Q&A prompt.
    
    Structure:
    - Answer summary (one sentence)
    - Detailed explanation (bullet points)
    - Related concepts (extended knowledge)
    - Practical applications (case studies)
    - References (citations)
    """
    context_text = "\n\n".join(
        f"[{i+1}] {(r.text or '').strip()[:500]}"
        for i, r in enumerate(contexts)
    )
    
    return [
        Message(
            role="system",
            content=(
                "你是一个专业的知识库问答助手。请严格基于提供的上下文内容回答用户问题。\n"
                "请按照以下结构组织你的回答：\n\n"
                "## 答案概要\n"
                "用一句话概括问题的核心答案。\n\n"
                "## 详细解释\n"
                "分点详细说明，每点包含：\n"
                "- 核心观点\n"
                "- 详细阐述\n"
                "- 关键细节\n\n"
                "## 相关概念\n"
                "介绍与问题相关的重要概念和知识：\n"
                "- 概念定义\n"
                "- 概念关系\n"
                "- 扩展知识\n\n"
                "## 实际应用\n"
                "提供具体的应用案例和场景：\n"
                "- 应用场景\n"
                "- 具体案例\n"
                "- 实践建议\n\n"
                "## 参考来源\n"
                "列出使用的参考资料，标注来源编号。\n\n"
                "如果上下文中没有足够信息，请明确告知用户，不要凭空捏造。"
            ),
        ),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n问题：{topic}",
        ),
    ]


def build_multi_perspective_qa_prompt(
    topic: str,
    contexts: List[Any],
    config: TemplateConfig,
    **kwargs
) -> List[Message]:
    """Build multi-perspective Q&A prompt.
    
    Perspectives:
    - Historical perspective
    - Scientific perspective
    - Application perspective
    - Controversy perspective
    - Frontier perspective
    """
    context_text = "\n\n".join(
        f"[{i+1}] {(r.text or '').strip()[:500]}"
        for i, r in enumerate(contexts)
    )
    
    return [
        Message(
            role="system",
            content=(
                "你是一个专业的知识库问答助手。请严格基于提供的上下文内容，从多个角度分析用户的问题。\n\n"
                "请按照以下结构组织你的回答：\n\n"
                "## 📜 历史视角\n"
                "从历史发展的角度分析：\n"
                "- 这个问题在历史上的演变过程\n"
                "- 关键的历史节点和事件\n"
                "- 历史背景对当前的影响\n\n"
                "## 🔬 科学视角\n"
                "从科学原理的角度分析：\n"
                "- 核心的科学原理和机制\n"
                "- 相关的理论基础\n"
                "- 科学研究的现状\n\n"
                "## 💡 应用视角\n"
                "从实际应用的角度分析：\n"
                "- 在现实中的应用场景\n"
                "- 具体的应用案例\n"
                "- 应用中的挑战和解决方案\n\n"
                "## ⚖️ 争议视角\n"
                "从争议讨论的角度分析：\n"
                "- 存在的不同观点和争议\n"
                "- 各方的主要论据\n"
                "- 争议的核心焦点\n\n"
                "## 🚀 前沿视角\n"
                "从前沿发展的角度分析：\n"
                "- 最新的研究进展\n"
                "- 未来的发展趋势\n"
                "- 潜在的突破方向\n\n"
                "如果上下文中没有足够信息，请明确告知用户，不要凭空捏造。"
            ),
        ),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n问题：{topic}",
        ),
    ]


def build_guided_qa_prompt(
    topic: str,
    contexts: List[Any],
    config: TemplateConfig,
    **kwargs
) -> List[Message]:
    """Build guided Q&A prompt.
    
    Structure:
    - Direct answer
    - Extended questions
    - Related topics
    - Learning path suggestions
    """
    context_text = "\n\n".join(
        f"[{i+1}] {(r.text or '').strip()[:500]}"
        for i, r in enumerate(contexts)
    )
    
    return [
        Message(
            role="system",
            content=(
                "你是一个专业的知识库问答助手。请严格基于提供的上下文内容回答用户问题，并引导用户深入思考。\n\n"
                "请按照以下结构组织你的回答：\n\n"
                "## 💭 直接回答\n"
                "首先直接回答用户的问题，提供清晰、准确的答案。\n\n"
                "## 🤔 延伸思考\n"
                "提出3-5个延伸问题，引导用户深入思考：\n"
                "- 问题1：[引导性问题]\n"
                "  - 思考方向：[提示思考方向]\n"
                "- 问题2：[引导性问题]\n"
                "  - 思考方向：[提示思考方向]\n"
                "- ...\n\n"
                "## 🔗 相关主题\n"
                "推荐相关的学习主题：\n"
                "- 相关主题1：[主题名称]\n"
                "  - 关联性：[说明为什么相关]\n"
                "- 相关主题2：[主题名称]\n"
                "  - 关联性：[说明为什么相关]\n"
                "- ...\n\n"
                "## 📚 学习路径\n"
                "提供学习路径建议：\n"
                "1. 基础知识：[建议先学习的基础内容]\n"
                "2. 进阶内容：[建议学习的进阶内容]\n"
                "3. 拓展延伸：[建议的拓展学习内容]\n\n"
                "如果上下文中没有足够信息，请明确告知用户，不要凭空捏造。"
            ),
        ),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n问题：{topic}",
        ),
    ]
