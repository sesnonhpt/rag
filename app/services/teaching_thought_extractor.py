"""
备课思路提取服务

从精品课文字稿中提取老师能理解的备课思路，而不是技术流程。
"""

from typing import List, Dict, Any, Optional
import json
import re
from dataclasses import dataclass, asdict

from src.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TeachingThought:
    """备课思路维度"""
    dimension: str  # what, how_intro, difficulties, activities, assessment
    icon: str  # emoji icon
    title: str
    content: str  # main description
    key_points: List[str]  # bullet points
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TeachingThoughtExtractor:
    """从精品课中提取备课思路"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def extract_thoughts(
        self,
        course_text: str,
        subject: str,
        topic: str,
        grade: str = "",
        teacher_name: str = ""
    ) -> List[TeachingThought]:
        """
        提取 5 个核心备课思路维度
        
        Args:
            course_text: 精品课文字稿
            subject: 学科（物理、化学等）
            topic: 课题（牛顿第一定律等）
            grade: 年级（可选）
            teacher_name: 主讲老师（可选）
        
        Returns:
            5 个备课思路维度的列表
        """
        logger.info(
            "teaching_thought_extractor.extract_start subject=%s topic=%s",
            subject, topic
        )
        
        try:
            # 构建提示词
            prompt = self._build_prompt(course_text, subject, topic, grade, teacher_name)
            
            # 调用 LLM（在后台线程中运行同步调用）
            import asyncio
            from src.libs.llm.base_llm import Message
            
            def _call_llm():
                return self.llm.chat([
                    Message(
                        role="system",
                        content="你是一位有 20 年教学经验的学科老师，擅长用简单的话解释备课思路。"
                    ),
                    Message(role="user", content=prompt)
                ])
            
            response = await asyncio.to_thread(_call_llm)
            
            # 解析返回内容
            thoughts = self._parse_response(response.content, topic, subject)
            
            logger.info(
                "teaching_thought_extractor.extract_success dimensions=%d",
                len(thoughts)
            )
            
            return thoughts
            
        except Exception as e:
            logger.exception("teaching_thought_extractor.extract_failed")
            # 降级：返回默认的思考框架
            return self._get_default_thoughts(topic, subject)
    
    def _build_prompt(
        self,
        course_text: str,
        subject: str,
        topic: str,
        grade: str,
        teacher_name: str
    ) -> str:
        """构建提示词"""
        
        # 截取前 3000 字（避免超过 token 限制）
        text_preview = course_text[:3000] if len(course_text) > 3000 else course_text
        
        teacher_info = f"主讲老师：{teacher_name}\n" if teacher_name else ""
        grade_info = f"年级：{grade}\n" if grade else ""
        
        prompt = f"""你是一位有 20 年教学经验的{subject}老师。现在有一位新老师要讲"{topic}"这节课，他拿到了名师的课堂实录，但不知道为什么要这样设计。

请用老师能听懂的话，从以下 5 个维度分析这节课的备课思路：

课程主题：{topic}
{grade_info}{teacher_info}
课程实录（前 3000 字）：
{text_preview}

---

请按以下格式分析（用老师的口吻，不要用教育学术语）：

## 1. 这节课要讲什么？
- 核心概念：[用一句话说清楚]
- 学生已经知道什么：[学情分析，2-3 句话]
- 这节课要让学生理解什么：[教学目标，用学生能听懂的话]

## 2. 怎么导入这个概念？
- 名师用了什么例子或情境：[具体描述]
- 为什么这样导入：[设计意图，2-3 句话]
- 其他班级可以怎么调整：[给出 1-2 个替代方案]

## 3. 学生容易卡在哪里？
- 易错点 1：[具体描述] → 突破策略：[怎么讲清楚]
- 易错点 2：[具体描述] → 突破策略：[怎么讲清楚]
- 易错点 3：[如果有的话]

## 4. 怎么设计课堂活动？
- 实验/讨论设计：[具体步骤，3-5 句话]
- 提问链：[引导学生思考的 3-5 个问题]
- 时间分配：[每个环节大概多久]

## 5. 怎么检验学生理解了？
- 当堂练习：[2-3 道题的类型和难度]
- 重点讲解：[哪道题需要详细讲，为什么]
- 课后作业：[如果需要的话]

---

注意：
- 每个维度用 3-5 个要点说清楚
- 用"学生容易混淆"而不是"教学难点"
- 用"怎么导入"而不是"情境创设"
- 用"检验理解"而不是"教学评价"
- 每个要点用一句话说清楚，不要太长
"""
        return prompt
    
    def _parse_response(self, content: str, topic: str, subject: str) -> List[TeachingThought]:
        """解析 LLM 返回的内容"""
        
        thoughts = []
        
        # 定义维度映射
        dimension_map = {
            "1. 这节课要讲什么": ("what_to_teach", "📚", "这节课要讲什么"),
            "2. 怎么导入这个概念": ("how_to_introduce", "🎯", "怎么导入"),
            "3. 学生容易卡在哪里": ("common_mistakes", "⚠️", "学生易错点"),
            "4. 怎么设计课堂活动": ("classroom_activities", "🎨", "课堂活动"),
            "5. 怎么检验学生理解了": ("check_understanding", "✅", "检验理解"),
        }
        
        # 按 ## 分割章节
        sections = content.split("## ")
        
        for section in sections[1:]:  # 跳过第一个空 section
            for key, (dimension, icon, title) in dimension_map.items():
                if section.startswith(key):
                    # 提取要点（以 - 或 • 开头的行）
                    lines = section.split("\n")
                    key_points = []
                    main_content = ""
                    
                    for line in lines[1:]:
                        line = line.strip()
                        if line and (line.startswith("-") or line.startswith("•")):
                            # 移除开头的 - 或 •
                            item = line.lstrip("-•").strip()
                            if item:
                                key_points.append(item)
                        elif line and not line.startswith("#"):
                            # 非标题行作为主要内容
                            if not main_content:
                                main_content = line
                    
                    # 如果没有提取到主要内容，使用第一个要点
                    if not main_content and key_points:
                        main_content = key_points[0]
                        key_points = key_points[1:]
                    
                    # 如果还是没有内容，使用默认文本
                    if not main_content:
                        main_content = f"正在分析{title}..."
                    
                    thoughts.append(TeachingThought(
                        dimension=dimension,
                        icon=icon,
                        title=title,
                        content=main_content,
                        key_points=key_points if key_points else [f"正在提取{title}的要点..."]
                    ))
                    break
        
        # 如果解析失败，返回默认框架
        if len(thoughts) < 5:
            logger.warning(
                "teaching_thought_extractor.parse_incomplete parsed=%d expected=5",
                len(thoughts)
            )
            return self._get_default_thoughts(topic, subject)
        
        return thoughts
    
    def _get_default_thoughts(self, topic: str, subject: str) -> List[TeachingThought]:
        """降级方案：返回默认的思考框架"""
        
        logger.info("teaching_thought_extractor.using_default_thoughts")
        
        return [
            TeachingThought(
                dimension="what_to_teach",
                icon="📚",
                title="这节课要讲什么",
                content=f"核心概念：{topic}",
                key_points=[
                    f"学生已经知道什么：正在分析{subject}课程的学情...",
                    "这节课要让学生理解什么：正在提取教学目标..."
                ]
            ),
            TeachingThought(
                dimension="how_to_introduce",
                icon="🎯",
                title="怎么导入",
                content="通过生活情境或实验现象引入",
                key_points=[
                    "名师用了什么例子：正在分析课堂实录...",
                    "为什么这样导入：正在提取设计意图...",
                    "其他班级可以怎么调整：正在生成替代方案..."
                ]
            ),
            TeachingThought(
                dimension="common_mistakes",
                icon="⚠️",
                title="学生易错点",
                content="容易混淆相关概念",
                key_points=[
                    "易错点 1：正在分析学生常见误解...",
                    "易错点 2：正在提取突破策略..."
                ]
            ),
            TeachingThought(
                dimension="classroom_activities",
                icon="🎨",
                title="课堂活动",
                content="设计动手实验、小组讨论",
                key_points=[
                    "实验观察：亲手操作",
                    "小组讨论：分享理解",
                    "问题探究：主动思考"
                ]
            ),
            TeachingThought(
                dimension="check_understanding",
                icon="✅",
                title="检验理解",
                content="通过课堂提问、练习题检验",
                key_points=[
                    "设计分层练习题",
                    "用提问检查深度",
                    "观察操作过程"
                ]
            ),
        ]


# 物理学科专用的提示词优化（可选）
class PhysicsTeachingThoughtExtractor(TeachingThoughtExtractor):
    """物理学科专用的备课思路提取器"""
    
    def _build_prompt(
        self,
        course_text: str,
        subject: str,
        topic: str,
        grade: str,
        teacher_name: str
    ) -> str:
        """物理学科的提示词优化"""
        
        base_prompt = super()._build_prompt(course_text, subject, topic, grade, teacher_name)
        
        # 增加物理学科特定的提示
        physics_hints = """

物理学科特别注意：
- 概念理解：物理概念往往反直觉，要特别关注学生的前概念
- 实验设计：物理实验是理解概念的关键，要详细描述实验步骤
- 数学推导：如果涉及公式，要说明推导思路，不要只给结果
- 生活联系：物理概念要联系生活实例，让学生感受到物理就在身边
"""
        
        return base_prompt + physics_hints
