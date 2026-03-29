"""
Subject-specific Templates for Modular RAG MCP Server.

This module provides templates tailored for different academic subjects.
"""

from typing import Any, List
from src.libs.llm.base_llm import Message
from . import TemplateConfig
from .base_sections import COMMON_SECTIONS


def build_physics_lesson_prompt(
    topic: str,
    contexts: List[Any],
    config: TemplateConfig,
    **kwargs
) -> List[Message]:
    """Build physics lesson plan prompt.
    
    Physics-specific elements:
    - Experimental design
    - Physical models
    - Mathematical derivations
    - Physical meaning
    - Common misconceptions
    """
    context_text = "\n\n".join(
        f"[{i+1}] {(r.text or '').strip()[:500]}"
        for i, r in enumerate(contexts)
    )
    
    instructions = []
    if config.include_background:
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
    
    if config.include_facts:
        instructions.append("""2. 核心知识点（详细列出）：
   - 基本概念：清晰定义该主题的核心概念和术语
   - 基本原理：详细阐述该主题的基本原理和规律
   - 重要公式：列出相关的数学公式和表达式
   - 实验现象：描述相关的实验现象和观测结果
   - 常见误区：指出学生容易产生的误解和错误认识
   
   **物理特色内容**：
   - 物理模型：介绍相关的理想化模型及其适用范围
   - 数学推导：详细推导重要公式的数学过程
   - 物理意义：深入解释概念的物理本质和意义
   - 量纲分析：介绍相关物理量的量纲和单位
   - 适用条件：明确理论或公式的适用条件和限制
   
   **相关人物与事件**：
   - 同时代科学家：介绍同时期的其他重要科学家及其贡献
   - 前驱工作：介绍该发现之前的相关研究和理论
   - 后续发展：介绍该发现之后的重要进展和突破
   - 跨学科影响：介绍该发现对其他学科领域的影响""")
    
    if config.include_examples:
        instructions.append("""3. 教学活动设计（具体可行）：
   - 演示实验：提供具体的实验设计，包括所需材料、操作步骤和预期结果
   - 课堂活动：设计互动性强的课堂活动，激发学生兴趣
   - 案例分析：提供真实的案例和应用场景
   - 思考问题：设计启发性的思考问题，引导学生深入思考
   - 拓展阅读：推荐相关的课外阅读材料和资源
   
   **物理实验设计**：
   - 实验目的：明确实验要验证或探究的物理规律
   - 实验原理：详细说明实验的理论依据
   - 实验器材：列出所需的实验器材和材料清单
   - 实验步骤：详细描述实验操作步骤
   - 数据记录：设计数据记录表格
   - 结果分析：说明如何分析实验数据
   - 注意事项：列出实验中的安全注意事项
   - 误差分析：分析可能的误差来源
   
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
                f"你是一名经验丰富的物理教师，擅长准备详细、深入的物理教案。请基于以下上下文内容，为主题'{topic}'生成一份详细的物理教案。\n"
                "请严格基于提供的上下文内容，不要凭空捏造信息。如果上下文中没有足够信息，请明确说明。\n\n"
                "请按照以下结构组织教案：\n\n"
                f"{instructions_text}\n\n"
                "4. 参考资源：列出使用的参考资料，包括来源文档和页码\n\n"
                "---\n\n"
                "**以下部分必须在所有教案中包含：**\n\n"
                f"{COMMON_SECTIONS}"
            ),
        ),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n请为'{topic}'生成一份详细的物理教案。",
        ),
    ]


def build_history_lesson_prompt(
    topic: str,
    contexts: List[Any],
    config: TemplateConfig,
    **kwargs
) -> List[Message]:
    """Build history lesson plan prompt.
    
    History-specific elements:
    - Historical background
    - Key figures
    - Event timeline
    - Historical significance
    - Historical controversies
    - Source analysis
    """
    context_text = "\n\n".join(
        f"[{i+1}] {(r.text or '').strip()[:500]}"
        for i, r in enumerate(contexts)
    )
    
    instructions = []
    if config.include_background:
        instructions.append("""1. 背景信息（详细展开）：
   - 时代背景：详细介绍该历史时期的时代特征、社会环境、政治经济状况
   - 历史背景：阐述该主题在历史长河中的位置和背景
   - 国际背景：介绍同时期国际社会的情况
   - 社会背景：说明当时的社会结构、文化特点、思想观念
   
   **关键人物（详细展开）**：
   - 人物生平：详细介绍关键人物的生平、出身、教育、经历
   - 历史角色：阐述该人物在历史事件中的作用和地位
   - 人物性格：分析人物的性格特点、思想观念、行为动机
   - 历史评价：介绍不同历史时期对该人物的评价
   - 相关人物：介绍与该人物相关的其他重要人物
   
   **事件背景（详细展开）**：
   - 事件起因：详细分析事件的深层原因和直接导火索
   - 相关事件：介绍与该事件相关的其他重要事件
   - 历史传统：介绍相关的历史传统和文化背景
   - 社会矛盾：分析当时存在的主要社会矛盾""")
    
    if config.include_facts:
        instructions.append("""2. 核心知识点（详细列出）：
   - 基本史实：清晰梳理该主题的基本历史事实
   - 时间线索：详细列出重要时间节点和事件顺序
   - 空间线索：说明事件发生的地理空间和范围
   - 因果关系：深入分析历史事件的因果关系
   - 历史意义：阐述该主题的历史意义和影响
   
   **历史特色内容**：
   - 史料分析：介绍相关的史料（一手史料、二手史料）
   - 史学观点：介绍不同的史学观点和解释
   - 历史争议：介绍存在的历史争议和不同看法
   - 历史教训：总结历史的经验教训
   - 现实启示：说明对现实的启示和借鉴意义
   
   **相关概念与理论**：
   - 历史概念：介绍相关的历史概念和术语
   - 历史理论：介绍相关的历史理论和分析方法
   - 历史方法：介绍历史研究的基本方法""")
    
    if config.include_examples:
        instructions.append("""3. 教学活动设计（具体可行）：
   - 史料研读：设计史料阅读和分析活动
   - 课堂讨论：设计启发性的讨论问题
   - 角色扮演：设计历史角色扮演活动
   - 案例分析：提供具体的历史案例
   - 拓展阅读：推荐相关的历史读物和资源
   
   **历史教学特色活动**：
   - 史料分析：指导学生分析一手史料和二手史料
   - 历史辩论：设计历史问题的辩论活动
   - 历史写作：指导学生进行历史小论文写作
   - 历史地图：使用历史地图分析事件
   - 历史时间线：构建历史事件的时间线
   - 历史对比：对比不同时期、不同地区的历史
   - 口述历史：收集和整理口述历史资料
   - 实地考察：推荐相关的历史遗迹、博物馆
   
   **思考与讨论**：
   - 历史假设：提出"如果...会怎样"的思考问题
   - 历史评价：引导学生评价历史人物和事件
   - 历史启示：引导学生思考历史的现实意义""")
    
    instructions_text = "\n\n".join(instructions)
    
    return [
        Message(
            role="system",
            content=(
                f"你是一名经验丰富的历史教师，擅长准备详细、深入的历史教案。请基于以下上下文内容，为主题'{topic}'生成一份详细的历史教案。\n"
                "请严格基于提供的上下文内容，不要凭空捏造信息。如果上下文中没有足够信息，请明确说明。\n\n"
                "请按照以下结构组织教案：\n\n"
                f"{instructions_text}\n\n"
                "4. 参考资源：列出使用的参考资料，包括来源文档和页码\n\n"
                "---\n\n"
                "**以下部分必须在所有教案中包含：**\n\n"
                f"{COMMON_SECTIONS}"
            ),
        ),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n请为'{topic}'生成一份详细的历史教案。",
        ),
    ]


def build_chinese_lesson_prompt(
    topic: str,
    contexts: List[Any],
    config: TemplateConfig,
    **kwargs
) -> List[Message]:
    """Build Chinese literature lesson plan prompt.
    
    Chinese-specific elements:
    - Author background
    - Work background
    - Text analysis
    - Thematic ideas
    - Artistic features
    - Related works
    """
    context_text = "\n\n".join(
        f"[{i+1}] {(r.text or '').strip()[:500]}"
        for i, r in enumerate(contexts)
    )
    
    instructions = []
    if config.include_background:
        instructions.append("""1. 背景信息（详细展开）：
   - 作者背景：详细介绍作者的生平、思想、创作风格、文学地位
   - 时代背景：介绍作品创作的时代背景和社会环境
   - 创作背景：说明作品的创作动机和创作过程
   - 文学背景：介绍当时的文学思潮和流派
   
   **作者详细介绍**：
   - 生平经历：作者的生平、重要经历、人生转折
   - 思想观念：作者的思想倾向、价值观念
   - 创作风格：作者的文学风格、艺术特色
   - 代表作品：作者的其他重要作品
   - 文学地位：作者在文学史上的地位和影响
   
   **作品背景详细介绍**：
   - 创作时间：作品创作的时间和时期
   - 创作动机：作者创作该作品的动机和目的
   - 社会环境：作品创作时的社会环境
   - 文学环境：作品创作时的文学环境""")
    
    if config.include_facts:
        instructions.append("""2. 核心知识点（详细列出）：
   - 文本分析：详细分析作品的结构、语言、修辞
   - 主题思想：深入阐述作品的主题思想和深层含义
   - 艺术特色：分析作品的艺术特色和表现手法
   - 人物形象：分析作品中的人物形象（如适用）
   - 情节结构：分析作品的情节结构（如适用）
   
   **文学特色内容**：
   - 语言特色：分析作品的语言特点和风格
   - 修辞手法：详细分析作品使用的修辞手法
   - 表现手法：分析作品的表现手法和技巧
   - 结构特点：分析作品的结构特点和布局
   - 意象分析：分析作品中的重要意象
   
   **相关作品与比较**：
   - 同期作品：介绍作者同时期的其他作品
   - 同类作品：介绍同类题材的其他作品
   - 对比分析：与其他作品的对比分析
   - 后世影响：作品对后世文学的影响""")
    
    if config.include_examples:
        instructions.append("""3. 教学活动设计（具体可行）：
   - 文本细读：设计文本细读和分析活动
   - 课堂讨论：设计启发性的讨论问题
   - 写作练习：设计相关的写作练习
   - 朗诵表演：设计朗诵或表演活动
   - 拓展阅读：推荐相关的阅读材料
   
   **语文教学特色活动**：
   - 文本解读：指导学生深入解读文本
   - 语言品味：引导学生品味语言之美
   - 意境赏析：指导学生赏析作品意境
   - 情感体验：引导学生体验作品情感
   - 写作训练：设计相关的写作训练
   - 对比阅读：设计对比阅读活动
   - 文学创作：指导学生进行文学创作
   - 戏剧表演：设计戏剧表演活动
   
   **思考与讨论**：
   - 深层思考：引导学生深入思考作品内涵
   - 个性化解读：鼓励学生进行个性化解读
   - 现实意义：引导学生思考作品的现实意义""")
    
    instructions_text = "\n\n".join(instructions)
    
    return [
        Message(
            role="system",
            content=(
                f"你是一名经验丰富的语文教师，擅长准备详细、深入的语文教案。请基于以下上下文内容，为主题'{topic}'生成一份详细的语文教案。\n"
                "请严格基于提供的上下文内容，不要凭空捏造信息。如果上下文中没有足够信息，请明确说明。\n\n"
                "请按照以下结构组织教案：\n\n"
                f"{instructions_text}\n\n"
                "4. 参考资源：列出使用的参考资料，包括来源文档和页码\n\n"
                "---\n\n"
                "**以下部分必须在所有教案中包含：**\n\n"
                f"{COMMON_SECTIONS}"
            ),
        ),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n请为'{topic}'生成一份详细的语文教案。",
        ),
    ]


def build_chemistry_lesson_prompt(
    topic: str,
    contexts: List[Any],
    config: TemplateConfig,
    **kwargs
) -> List[Message]:
    """Build chemistry lesson plan prompt."""
    context_text = "\n\n".join(
        f"[{i+1}] {(r.text or '').strip()[:500]}"
        for i, r in enumerate(contexts)
    )
    
    instructions = []
    if config.include_background:
        instructions.append("""1. 背景信息（详细展开）：
   - 历史背景：详细介绍该主题的历史发展脉络
   - 科学意义：阐述该主题在化学史上的重要地位
   - 现实应用：说明该主题在现代社会的应用价值
   - 相关概念：介绍与该主题相关的其他重要概念
   
   **人物背景（如适用）**：
   - 生平简介：详细介绍相关化学家的生平和成就
   - 研究历程：详细描述研究过程和突破
   - 历史评价：介绍该化学家的历史地位""")
    
    if config.include_facts:
        instructions.append("""2. 核心知识点（详细列出）：
   - 基本概念：清晰定义该主题的核心概念和术语
   - 反应机理：详细阐述化学反应的机理和过程
   - 化学方程式：列出相关的化学方程式
   - 实验现象：描述相关的实验现象
   - 常见误区：指出学生容易产生的误解
   
   **化学特色内容**：
   - 反应类型：介绍化学反应的类型和特点
   - 反应条件：说明反应所需的条件
   - 能量变化：分析反应中的能量变化
   - 平衡移动：介绍化学平衡的移动
   - 安全注意事项：列出安全注意事项""")
    
    if config.include_examples:
        instructions.append("""3. 教学活动设计（具体可行）：
   - 演示实验：提供具体的实验设计
   - 课堂活动：设计互动性强的课堂活动
   - 案例分析：提供真实的案例和应用场景
   - 思考问题：设计启发性的思考问题
   - 拓展阅读：推荐相关的课外阅读材料
   
   **化学实验设计**：
   - 实验目的：明确实验目的
   - 实验原理：详细说明实验原理
   - 实验器材：列出所需的实验器材
   - 实验步骤：详细描述实验步骤
   - 数据记录：设计数据记录表格
   - 结果分析：说明如何分析结果
   - 安全事项：列出安全注意事项""")
    
    instructions_text = "\n\n".join(instructions)
    
    return [
        Message(
            role="system",
            content=(
                f"你是一名经验丰富的化学教师，擅长准备详细、深入的化学教案。请基于以下上下文内容，为主题'{topic}'生成一份详细的化学教案。\n"
                "请严格基于提供的上下文内容，不要凭空捏造信息。如果上下文中没有足够信息，请明确说明。\n\n"
                "请按照以下结构组织教案：\n\n"
                f"{instructions_text}\n\n"
                "4. 参考资源：列出使用的参考资料，包括来源文档和页码\n\n"
                "---\n\n"
                "**以下部分必须在所有教案中包含：**\n\n"
                f"{COMMON_SECTIONS}"
            ),
        ),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n请为'{topic}'生成一份详细的化学教案。",
        ),
    ]


def build_biology_lesson_prompt(
    topic: str,
    contexts: List[Any],
    config: TemplateConfig,
    **kwargs
) -> List[Message]:
    """Build biology lesson plan prompt."""
    context_text = "\n\n".join(
        f"[{i+1}] {(r.text or '').strip()[:500]}"
        for i, r in enumerate(contexts)
    )
    
    instructions = []
    if config.include_background:
        instructions.append("""1. 背景信息（详细展开）：
   - 历史背景：详细介绍该主题的历史发展脉络
   - 科学意义：阐述该主题在生物学史上的重要地位
   - 现实应用：说明该主题在现代社会的应用价值
   - 相关概念：介绍与该主题相关的其他重要概念
   
   **人物背景（如适用）**：
   - 生平简介：详细介绍相关生物学家的生平和成就
   - 研究历程：详细描述研究过程和突破
   - 历史评价：介绍该生物学家的历史地位""")
    
    if config.include_facts:
        instructions.append("""2. 核心知识点（详细列出）：
   - 基本概念：清晰定义该主题的核心概念和术语
   - 生命现象：详细阐述相关的生命现象和过程
   - 结构功能：介绍生物体的结构和功能
   - 实验现象：描述相关的实验现象
   - 常见误区：指出学生容易产生的误解
   
   **生物特色内容**：
   - 分类系统：介绍生物的分类系统
   - 生命活动：介绍生命活动的过程
   - 生态意义：阐述生态学意义
   - 进化视角：从进化角度分析
   - 现代技术：介绍相关的现代生物技术""")
    
    if config.include_examples:
        instructions.append("""3. 教学活动设计（具体可行）：
   - 演示实验：提供具体的实验设计
   - 课堂活动：设计互动性强的课堂活动
   - 案例分析：提供真实的案例和应用场景
   - 思考问题：设计启发性的思考问题
   - 拓展阅读：推荐相关的课外阅读材料
   
   **生物实验设计**：
   - 实验目的：明确实验目的
   - 实验原理：详细说明实验原理
   - 实验器材：列出所需的实验器材
   - 实验步骤：详细描述实验步骤
   - 数据记录：设计数据记录表格
   - 结果分析：说明如何分析结果
   - 注意事项：列出注意事项""")
    
    instructions_text = "\n\n".join(instructions)
    
    return [
        Message(
            role="system",
            content=(
                f"你是一名经验丰富的生物教师，擅长准备详细、深入的生物教案。请基于以下上下文内容，为主题'{topic}'生成一份详细的生物教案。\n"
                "请严格基于提供的上下文内容，不要凭空捏造信息。如果上下文中没有足够信息，请明确说明。\n\n"
                "请按照以下结构组织教案：\n\n"
                f"{instructions_text}\n\n"
                "4. 参考资源：列出使用的参考资料，包括来源文档和页码\n\n"
                "---\n\n"
                "**以下部分必须在所有教案中包含：**\n\n"
                f"{COMMON_SECTIONS}"
            ),
        ),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n请为'{topic}'生成一份详细的生物教案。",
        ),
    ]


def build_math_lesson_prompt(
    topic: str,
    contexts: List[Any],
    config: TemplateConfig,
    **kwargs
) -> List[Message]:
    """Build mathematics lesson plan prompt."""
    context_text = "\n\n".join(
        f"[{i+1}] {(r.text or '').strip()[:500]}"
        for i, r in enumerate(contexts)
    )
    
    instructions = []
    if config.include_background:
        instructions.append("""1. 背景信息（详细展开）：
   - 历史背景：详细介绍该主题的历史发展脉络
   - 数学意义：阐述该主题在数学史上的重要地位
   - 现实应用：说明该主题在现代社会的应用价值
   - 相关概念：介绍与该主题相关的其他重要概念
   
   **人物背景（如适用）**：
   - 生平简介：详细介绍相关数学家的生平和成就
   - 研究历程：详细描述研究过程和突破
   - 历史评价：介绍该数学家的历史地位""")
    
    if config.include_facts:
        instructions.append("""2. 核心知识点（详细列出）：
   - 基本概念：清晰定义该主题的核心概念和术语
   - 定理公式：详细阐述相关的定理和公式
   - 证明过程：提供定理或公式的证明过程
   - 典型例题：提供典型的例题和解法
   - 常见误区：指出学生容易产生的误解
   
   **数学特色内容**：
   - 数学思想：介绍相关的数学思想方法
   - 解题方法：介绍解题的方法和技巧
   - 推广应用：介绍定理或公式的推广应用
   - 数学美：展示数学的美学价值
   - 数学建模：介绍相关的数学建模""")
    
    if config.include_examples:
        instructions.append("""3. 教学活动设计（具体可行）：
   - 概念引入：设计概念引入的活动
   - 定理证明：设计定理证明的教学活动
   - 例题讲解：提供典型的例题讲解
   - 练习设计：设计分层练习题
   - 拓展阅读：推荐相关的课外阅读材料
   
   **数学教学特色活动**：
   - 问题情境：创设问题情境
   - 探究活动：设计数学探究活动
   - 合作学习：设计小组合作学习
   - 数学实验：设计数学实验活动
   - 数学建模：设计数学建模活动
   
   **练习设计**：
   - 基础练习：设计基础练习题
   - 提高练习：设计提高练习题
   - 拓展练习：设计拓展练习题
   - 应用练习：设计应用练习题""")
    
    instructions_text = "\n\n".join(instructions)
    
    return [
        Message(
            role="system",
            content=(
                f"你是一名经验丰富的数学教师，擅长准备详细、深入的数学教案。请基于以下上下文内容，为主题'{topic}'生成一份详细的数学教案。\n"
                "请严格基于提供的上下文内容，不要凭空捏造信息。如果上下文中没有足够信息，请明确说明。\n\n"
                "请按照以下结构组织教案：\n\n"
                f"{instructions_text}\n\n"
                "4. 参考资源：列出使用的参考资料，包括来源文档和页码\n\n"
                "---\n\n"
                "**以下部分必须在所有教案中包含：**\n\n"
                f"{COMMON_SECTIONS}"
            ),
        ),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n请为'{topic}'生成一份详细的数学教案。",
        ),
    ]
