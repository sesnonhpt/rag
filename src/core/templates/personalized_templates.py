"""
Personalized Templates for Modular RAG MCP Server.

This module provides templates personalized for different learner characteristics.
"""

from typing import Any, List
from src.libs.llm.base_llm import Message
from . import TemplateConfig, GradeLevel, LearningStyle
from .base_sections import COMMON_SECTIONS


def build_grade_personalized_prompt(
    topic: str,
    contexts: List[Any],
    config: TemplateConfig,
    **kwargs
) -> List[Message]:
    """Build grade-level personalized prompt.
    
    Features:
    - Adjust content depth based on grade level
    - Adjust language complexity
    - Adjust activity design
    - Adjust learning expectations
    """
    context_text = "\n\n".join(
        f"[{i+1}] {(r.text or '').strip()[:500]}"
        for i, r in enumerate(contexts)
    )
    
    grade_level = config.grade_level if config.grade_level else GradeLevel.MIDDLE
    
    grade_characteristics = {
        GradeLevel.PRIMARY: {
            "name": "小学",
            "language_style": "生动活泼，故事性强，使用简单的语言和形象的比喻",
            "content_depth": "基础概念，直观理解，避免抽象理论",
            "activity_design": "游戏化学习，动手实践，角色扮演",
            "time_attention": "注意力集中时间短，活动要丰富多样",
            "examples": "使用生活中的实例，贴近学生的日常经验",
        },
        GradeLevel.MIDDLE: {
            "name": "初中",
            "language_style": "清晰准确，逻辑性强，适当使用专业术语",
            "content_depth": "概念清晰，原理明确，初步的抽象思维",
            "activity_design": "探究实验，小组讨论，案例分析",
            "time_attention": "注意力集中时间中等，可以有一定深度的讨论",
            "examples": "结合实际应用，展示知识的实用价值",
        },
        GradeLevel.HIGH: {
            "name": "高中",
            "language_style": "严谨专业，逻辑严密，使用专业术语",
            "content_depth": "深度分析，抽象思维，理论推导",
            "activity_design": "研究性学习，深度讨论，项目式学习",
            "time_attention": "注意力集中时间长，可以进行深度学习",
            "examples": "结合前沿研究，展示知识的科学价值",
        },
        GradeLevel.COLLEGE: {
            "name": "大学",
            "language_style": "学术严谨，专业深入，使用学术语言",
            "content_depth": "学术研究，批判思维，创新探索",
            "activity_design": "学术研究，创新实践，学术交流",
            "time_attention": "自主学习能力强，可以进行独立研究",
            "examples": "结合学术研究，展示知识的前沿发展",
        },
    }
    
    characteristics = grade_characteristics[grade_level]
    
    return [
        Message(
            role="system",
            content=(
                f"你是一名经验丰富的{characteristics['name']}教师，擅长根据学生的年级特点设计教学内容。\n"
                f"请基于以下上下文内容，为主题'{topic}'生成一份适合{characteristics['name']}学生的教案。\n"
                "请严格基于提供的上下文内容，不要凭空捏造信息。如果上下文中没有足够信息，请明确说明。\n\n"
                f"## 年级特点\n"
                f"**年级**：{characteristics['name']}\n\n"
                f"**语言风格**：{characteristics['language_style']}\n\n"
                f"**内容深度**：{characteristics['content_depth']}\n\n"
                f"**活动设计**：{characteristics['activity_design']}\n\n"
                f"**注意力特点**：{characteristics['time_attention']}\n\n"
                f"**案例选择**：{characteristics['examples']}\n\n"
                "请按照以下结构组织教案：\n\n"
                "## 🎯 教学目标\n"
                "根据年级特点设定教学目标：\n\n"
                "### 知识目标\n"
                "- [适合该年级的知识目标]\n\n"
                "### 能力目标\n"
                "- [适合该年级的能力目标]\n\n"
                "### 情感目标\n"
                "- [适合该年级的情感目标]\n\n"
                "## 📚 教学内容\n"
                "根据年级特点调整内容深度：\n\n"
                "### 核心概念\n"
                "- 概念1：[用适合该年级的语言解释]\n"
                "- 概念2：[用适合该年级的语言解释]\n\n"
                "### 重点内容\n"
                "- 重点1：[适合该年级的重点内容]\n"
                "- 重点2：[适合该年级的重点内容]\n\n"
                "### 难点突破\n"
                "- 难点1：[如何突破，适合该年级的方法]\n"
                "- 难点2：[如何突破，适合该年级的方法]\n\n"
                "## 🎨 教学活动\n"
                "根据年级特点设计教学活动：\n\n"
                "### 导入活动\n"
                "**活动名称**：[活动名称]\n\n"
                "**活动形式**：[适合该年级的活动形式]\n\n"
                "**活动步骤**：\n"
                "1. 步骤1：[具体步骤]\n"
                "2. 步骤2：[具体步骤]\n"
                "3. 步骤3：[具体步骤]\n\n"
                "**设计意图**：[为什么要这样设计]\n\n"
                "### 主体活动\n"
                "**活动名称**：[活动名称]\n\n"
                "**活动形式**：[适合该年级的活动形式]\n\n"
                "**活动步骤**：\n"
                "1. 步骤1：[具体步骤]\n"
                "2. 步骤2：[具体步骤]\n"
                "3. 步骤3：[具体步骤]\n\n"
                "**设计意图**：[为什么要这样设计]\n\n"
                "### 总结活动\n"
                "**活动名称**：[活动名称]\n\n"
                "**活动形式**：[适合该年级的活动形式]\n\n"
                "**活动步骤**：\n"
                "1. 步骤1：[具体步骤]\n"
                "2. 步骤2：[具体步骤]\n\n"
                "**设计意图**：[为什么要这样设计]\n\n"
                "## 📝 教学案例\n"
                "选择适合该年级的案例：\n\n"
                "### 案例1\n"
                "**案例内容**：[适合该年级的案例]\n\n"
                "**案例特点**：[为什么适合该年级]\n\n"
                "**教学使用**：[如何使用这个案例]\n\n"
                "### 案例2\n"
                "**案例内容**：[适合该年级的案例]\n\n"
                "**案例特点**：[为什么适合该年级]\n\n"
                "**教学使用**：[如何使用这个案例]\n\n"
                "## 💬 语言表达\n"
                "使用适合该年级的语言：\n\n"
                "### 概念解释\n"
                "- 原始概念：[专业术语]\n"
                "- 学生语言：[适合该年级的解释]\n\n"
                "### 问题提问\n"
                "- 提问方式：[适合该年级的提问方式]\n"
                "- 问题示例：[具体问题]\n\n"
                "## ⏰ 时间安排\n"
                "根据注意力特点安排时间：\n\n"
                "| 环节 | 时间 | 活动 | 设计理由 |\n"
                "|------|------|------|----------|\n"
                "| 导入 | [时间] | [活动] | [为什么这样安排] |\n"
                "| 新授 | [时间] | [活动] | [为什么这样安排] |\n"
                "| 练习 | [时间] | [活动] | [为什么这样安排] |\n"
                "| 总结 | [时间] | [活动] | [为什么这样安排] |\n\n"
                "## 📊 评价方式\n"
                "设计适合该年级的评价方式：\n\n"
                "### 过程评价\n"
                "- 评价方式1：[适合该年级的评价方式]\n"
                "- 评价方式2：[适合该年级的评价方式]\n\n"
                "### 结果评价\n"
                "- 评价方式1：[适合该年级的评价方式]\n"
                "- 评价方式2：[适合该年级的评价方式]\n\n"
                "## 📚 参考资源\n"
                "列出使用的参考资料，包括来源文档和页码\n\n"
                "---\n\n"
                "**以下部分必须在所有教案中包含：**\n\n"
                f"{COMMON_SECTIONS}"
            ),
        ),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n请为{characteristics['name']}学生生成教案。",
        ),
    ]


def build_style_personalized_prompt(
    topic: str,
    contexts: List[Any],
    config: TemplateConfig,
    **kwargs
) -> List[Message]:
    """Build learning style personalized prompt.
    
    Features:
    - Adjust content presentation based on learning style
    - Provide multiple representation methods
    - Design style-specific activities
    - Offer personalized learning suggestions
    """
    context_text = "\n\n".join(
        f"[{i+1}] {(r.text or '').strip()[:500]}"
        for i, r in enumerate(contexts)
    )
    
    learning_style = config.learning_style if config.learning_style else LearningStyle.VISUAL
    
    style_characteristics = {
        LearningStyle.VISUAL: {
            "name": "视觉型",
            "description": "通过图像、图表、视频等视觉方式学习效果最好",
            "presentation": "使用图表、思维导图、流程图、图片、视频等视觉元素",
            "activities": "观看视频、绘制图表、制作思维导图、观察实验",
            "materials": "图表、图片、视频、模型、实物展示",
            "strategies": "用颜色标注重点、画图理解概念、制作知识卡片",
        },
        LearningStyle.AUDITORY: {
            "name": "听觉型",
            "description": "通过听讲、讨论、音频等听觉方式学习效果最好",
            "presentation": "使用口头讲解、音频材料、讨论交流、朗读背诵",
            "activities": "听讲座、小组讨论、朗读、辩论、口头报告",
            "materials": "音频材料、讲座视频、讨论话题、朗读材料",
            "strategies": "朗读记忆、录音复习、参加讨论、口头复述",
        },
        LearningStyle.KINESTHETIC: {
            "name": "动觉型",
            "description": "通过动手操作、实践活动、身体参与学习效果最好",
            "presentation": "使用实验操作、动手制作、角色扮演、实地考察",
            "activities": "实验操作、手工制作、角色扮演、实地考察、游戏活动",
            "materials": "实验器材、制作材料、道具、实地考察资源",
            "strategies": "动手实践、做中学、身体记忆、实地体验",
        },
        LearningStyle.READ_WRITE: {
            "name": "读写型",
            "description": "通过阅读文本、写作笔记等读写方式学习效果最好",
            "presentation": "使用文本材料、笔记整理、写作练习、阅读理解",
            "activities": "阅读文本、做笔记、写作练习、制作提纲、整理资料",
            "materials": "文本材料、参考书籍、笔记本、写作工具",
            "strategies": "做笔记、写总结、制作提纲、阅读理解",
        },
    }
    
    characteristics = style_characteristics[learning_style]
    
    return [
        Message(
            role="system",
            content=(
                f"你是一名擅长个性化教学的教师，能够根据学生的学习风格调整教学方式。\n"
                f"请基于以下上下文内容，为主题'{topic}'生成一份适合{characteristics['name']}学习者的教案。\n"
                "请严格基于提供的上下文内容，不要凭空捏造信息。如果上下文中没有足够信息，请明确说明。\n\n"
                f"## 学习风格特点\n"
                f"**学习风格**：{characteristics['name']}\n\n"
                f"**特点描述**：{characteristics['description']}\n\n"
                f"**呈现方式**：{characteristics['presentation']}\n\n"
                f"**活动设计**：{characteristics['activities']}\n\n"
                f"**教学材料**：{characteristics['materials']}\n\n"
                f"**学习策略**：{characteristics['strategies']}\n\n"
                "请按照以下结构组织教案：\n\n"
                "## 🎯 教学目标\n"
                "设定适合该学习风格的教学目标：\n\n"
                "### 知识目标\n"
                "- [知识目标]\n\n"
                "### 能力目标\n"
                "- [能力目标，强调该学习风格相关的能力]\n\n"
                "### 情感目标\n"
                "- [情感目标]\n\n"
                "## 📚 内容呈现\n"
                "使用适合该学习风格的方式呈现内容：\n\n"
                "### 核心概念\n"
                "用适合该学习风格的方式呈现核心概念：\n\n"
                "#### 概念1：[概念名称]\n"
                "**呈现方式**：[如何用该学习风格的方式呈现]\n\n"
                "**具体内容**：[具体的内容呈现]\n\n"
                "**辅助材料**：[需要的辅助材料]\n\n"
                "#### 概念2：[概念名称]\n"
                "**呈现方式**：[如何用该学习风格的方式呈现]\n\n"
                "**具体内容**：[具体的内容呈现]\n\n"
                "**辅助材料**：[需要的辅助材料]\n\n"
                "### 知识结构\n"
                "用适合该学习风格的方式展示知识结构：\n\n"
                "**结构呈现**：[如何用该学习风格的方式展示结构]\n\n"
                "**具体内容**：[具体的结构展示]\n\n"
                "## 🎨 教学活动\n"
                "设计适合该学习风格的教学活动：\n\n"
                "### 活动1：[活动名称]\n"
                "**活动类型**：[适合该学习风格的活动类型]\n\n"
                "**活动目标**：[活动要达到的目标]\n\n"
                "**活动材料**：[需要的材料]\n\n"
                "**活动步骤**：\n"
                "1. 步骤1：[具体步骤]\n"
                "2. 步骤2：[具体步骤]\n"
                "3. 步骤3：[具体步骤]\n\n"
                "**设计理由**：[为什么适合该学习风格]\n\n"
                "### 活动2：[活动名称]\n"
                "**活动类型**：[适合该学习风格的活动类型]\n\n"
                "**活动目标**：[活动要达到的目标]\n\n"
                "**活动材料**：[需要的材料]\n\n"
                "**活动步骤**：\n"
                "1. 步骤1：[具体步骤]\n"
                "2. 步骤2：[具体步骤]\n"
                "3. 步骤3：[具体步骤]\n\n"
                "**设计理由**：[为什么适合该学习风格]\n\n"
                "## 📦 教学材料\n"
                "准备适合该学习风格的教学材料：\n\n"
                "### 主要材料\n"
                "- 材料1：[具体材料，适合该学习风格]\n"
                "- 材料2：[具体材料，适合该学习风格]\n"
                "- 材料3：[具体材料，适合该学习风格]\n\n"
                "### 辅助材料\n"
                "- 辅助材料1：[具体材料]\n"
                "- 辅助材料2：[具体材料]\n\n"
                "## 💡 学习策略\n"
                "提供适合该学习风格的学习策略：\n\n"
                "### 理解策略\n"
                "- 策略1：[如何用该学习风格的方式理解]\n"
                "- 策略2：[如何用该学习风格的方式理解]\n\n"
                "### 记忆策略\n"
                "- 策略1：[如何用该学习风格的方式记忆]\n"
                "- 策略2：[如何用该学习风格的方式记忆]\n\n"
                "### 应用策略\n"
                "- 策略1：[如何用该学习风格的方式应用]\n"
                "- 策略2：[如何用该学习风格的方式应用]\n\n"
                "## 🔄 多元呈现\n"
                "虽然针对特定学习风格，但也提供其他方式的呈现：\n\n"
                "### 视觉呈现\n"
                "- 呈现方式：[视觉呈现方式]\n"
                "- 适用内容：[哪些内容适合视觉呈现]\n\n"
                "### 听觉呈现\n"
                "- 呈现方式：[听觉呈现方式]\n"
                "- 适用内容：[哪些内容适合听觉呈现]\n\n"
                "### 动觉呈现\n"
                "- 呈现方式：[动觉呈现方式]\n"
                "- 适用内容：[哪些内容适合动觉呈现]\n\n"
                "### 读写呈现\n"
                "- 呈现方式：[读写呈现方式]\n"
                "- 适用内容：[哪些内容适合读写呈现]\n\n"
                "## 📊 评价方式\n"
                "设计适合该学习风格的评价方式：\n\n"
                "### 过程评价\n"
                "- 评价方式1：[适合该学习风格的评价方式]\n"
                "- 评价方式2：[适合该学习风格的评价方式]\n\n"
                "### 结果评价\n"
                "- 评价方式1：[适合该学习风格的评价方式]\n"
                "- 评价方式2：[适合该学习风格的评价方式]\n\n"
                "## 📚 参考资源\n"
                "列出使用的参考资料，包括来源文档和页码\n\n"
                "---\n\n"
                "**以下部分必须在所有教案中包含：**\n\n"
                f"{COMMON_SECTIONS}"
            ),
        ),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n请为{characteristics['name']}学习者生成教案。",
        ),
    ]
