"""
Template Manager for Modular RAG MCP Server.

This module provides a comprehensive template management system for generating
various types of educational content, including lesson plans, Q&A, exercises,
and interactive learning materials.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json

from src.libs.llm.base_llm import Message


class TemplateType(Enum):
    """Template types for different use cases."""
    
    # Q&A Templates
    QA_STRUCTURED = "qa_structured"
    QA_MULTI_PERSPECTIVE = "qa_multi_perspective"
    QA_GUIDED = "qa_guided"
    
    # Subject-specific Templates
    SUBJECT_PHYSICS = "subject_physics"
    SUBJECT_HISTORY = "subject_history"
    SUBJECT_CHINESE = "subject_chinese"
    SUBJECT_CHEMISTRY = "subject_chemistry"
    SUBJECT_BIOLOGY = "subject_biology"
    SUBJECT_MATH = "subject_math"
    SUBJECT_ENGLISH = "subject_english"
    SUBJECT_GEOGRAPHY = "subject_geography"

    # Guide Templates
    GUIDE_MATH = "guide_math"
    GUIDE_PHYSICS = "guide_physics"
    GUIDE_CHEMISTRY = "guide_chemistry"
    GUIDE_BIOLOGY = "guide_biology"
    GUIDE_HISTORY = "guide_history"
    GUIDE_GEOGRAPHY = "guide_geography"
    GUIDE_CHINESE = "guide_chinese"
    GUIDE_ENGLISH = "guide_english"
    GUIDE_POLITICS = "guide_politics"
    GUIDE_MASTER = "guide_master"

    # Comprehensive Template
    COMPREHENSIVE_MASTER = "comprehensive_master"
    
    # Lesson Type Templates
    LESSON_NEW = "lesson_new"
    LESSON_REVIEW = "lesson_review"
    LESSON_EXPERIMENT = "lesson_experiment"
    LESSON_DISCUSSION = "lesson_discussion"
    
    # Function Templates
    FUNC_EXERCISE = "func_exercise"
    FUNC_NOTES = "func_notes"
    FUNC_SUMMARY = "func_summary"
    FUNC_LEARNING_PATH = "func_learning_path"
    
    # Interactive Templates
    INTERACTIVE_SOCRATIC = "interactive_socratic"
    INTERACTIVE_DEBATE = "interactive_debate"
    INTERACTIVE_PROJECT = "interactive_project"
    
    # Personalized Templates
    PERSONALIZED_GRADE = "personalized_grade"
    PERSONALIZED_STYLE = "personalized_style"


class GradeLevel(Enum):
    """Grade levels for personalized templates."""
    PRIMARY = "primary"  # 小学
    MIDDLE = "middle"    # 初中
    HIGH = "high"        # 高中
    COLLEGE = "college"  # 大学


class LearningStyle(Enum):
    """Learning styles for personalized templates."""
    VISUAL = "visual"      # 视觉型
    AUDITORY = "auditory"  # 听觉型
    KINESTHETIC = "kinesthetic"  # 动觉型
    READ_WRITE = "read_write"    # 读写型


@dataclass
class TemplateConfig:
    """Configuration for template generation."""
    template_type: TemplateType
    grade_level: Optional[GradeLevel] = None
    learning_style: Optional[LearningStyle] = None
    include_background: bool = True
    include_facts: bool = True
    include_examples: bool = True
    include_exercises: bool = False
    difficulty: str = "medium"  # easy, medium, hard
    language: str = "zh"  # zh, en
    custom_instructions: Optional[str] = None


class TemplateManager:
    """Manager for all template types."""
    
    def __init__(self):
        self.templates: Dict[TemplateType, callable] = {}
        self._register_templates()
    
    def _register_templates(self):
        """Register all available templates."""
        from .qa_templates import (
            build_structured_qa_prompt,
            build_multi_perspective_qa_prompt,
            build_guided_qa_prompt,
        )
        from .subject_templates import (
            build_physics_lesson_prompt,
            build_history_lesson_prompt,
            build_chinese_lesson_prompt,
            build_chemistry_lesson_prompt,
            build_biology_lesson_prompt,
            build_math_lesson_prompt,
        )
        from .lesson_type_templates import (
            build_new_lesson_prompt,
            build_review_lesson_prompt,
            build_experiment_lesson_prompt,
            build_discussion_lesson_prompt,
        )
        from .guide_templates import (
            build_guide_master_prompt,
            build_math_guide_prompt,
            build_physics_guide_prompt,
            build_chemistry_guide_prompt,
            build_biology_guide_prompt,
            build_history_guide_prompt,
            build_geography_guide_prompt,
            build_chinese_guide_prompt,
            build_english_guide_prompt,
            build_politics_guide_prompt,
        )
        from .comprehensive_templates import build_comprehensive_master_prompt
        from .function_templates import (
            build_exercise_prompt,
            build_notes_prompt,
            build_summary_prompt,
            build_learning_path_prompt,
        )
        from .interactive_templates import (
            build_socratic_prompt,
            build_debate_prompt,
            build_project_prompt,
        )
        from .personalized_templates import (
            build_grade_personalized_prompt,
            build_style_personalized_prompt,
        )
        
        # Q&A Templates
        self.templates[TemplateType.QA_STRUCTURED] = build_structured_qa_prompt
        self.templates[TemplateType.QA_MULTI_PERSPECTIVE] = build_multi_perspective_qa_prompt
        self.templates[TemplateType.QA_GUIDED] = build_guided_qa_prompt
        
        # Subject Templates
        self.templates[TemplateType.SUBJECT_PHYSICS] = build_physics_lesson_prompt
        self.templates[TemplateType.SUBJECT_HISTORY] = build_history_lesson_prompt
        self.templates[TemplateType.SUBJECT_CHINESE] = build_chinese_lesson_prompt
        self.templates[TemplateType.SUBJECT_CHEMISTRY] = build_chemistry_lesson_prompt
        self.templates[TemplateType.SUBJECT_BIOLOGY] = build_biology_lesson_prompt
        self.templates[TemplateType.SUBJECT_MATH] = build_math_lesson_prompt

        # Guide Templates
        self.templates[TemplateType.GUIDE_MASTER] = build_guide_master_prompt
        self.templates[TemplateType.GUIDE_MATH] = build_math_guide_prompt
        self.templates[TemplateType.GUIDE_PHYSICS] = build_physics_guide_prompt
        self.templates[TemplateType.GUIDE_CHEMISTRY] = build_chemistry_guide_prompt
        self.templates[TemplateType.GUIDE_BIOLOGY] = build_biology_guide_prompt
        self.templates[TemplateType.GUIDE_HISTORY] = build_history_guide_prompt
        self.templates[TemplateType.GUIDE_GEOGRAPHY] = build_geography_guide_prompt
        self.templates[TemplateType.GUIDE_CHINESE] = build_chinese_guide_prompt
        self.templates[TemplateType.GUIDE_ENGLISH] = build_english_guide_prompt
        self.templates[TemplateType.GUIDE_POLITICS] = build_politics_guide_prompt

        # Comprehensive Template
        self.templates[TemplateType.COMPREHENSIVE_MASTER] = build_comprehensive_master_prompt
        
        # Lesson Type Templates
        self.templates[TemplateType.LESSON_NEW] = build_new_lesson_prompt
        self.templates[TemplateType.LESSON_REVIEW] = build_review_lesson_prompt
        self.templates[TemplateType.LESSON_EXPERIMENT] = build_experiment_lesson_prompt
        self.templates[TemplateType.LESSON_DISCUSSION] = build_discussion_lesson_prompt
        
        # Function Templates
        self.templates[TemplateType.FUNC_EXERCISE] = build_exercise_prompt
        self.templates[TemplateType.FUNC_NOTES] = build_notes_prompt
        self.templates[TemplateType.FUNC_SUMMARY] = build_summary_prompt
        self.templates[TemplateType.FUNC_LEARNING_PATH] = build_learning_path_prompt
        
        # Interactive Templates
        self.templates[TemplateType.INTERACTIVE_SOCRATIC] = build_socratic_prompt
        self.templates[TemplateType.INTERACTIVE_DEBATE] = build_debate_prompt
        self.templates[TemplateType.INTERACTIVE_PROJECT] = build_project_prompt
        
        # Personalized Templates
        self.templates[TemplateType.PERSONALIZED_GRADE] = build_grade_personalized_prompt
        self.templates[TemplateType.PERSONALIZED_STYLE] = build_style_personalized_prompt
    
    def build_prompt(
        self,
        config: TemplateConfig,
        topic: str,
        contexts: List[Any],
        **kwargs
    ) -> List[Message]:
        """Build prompt using specified template.
        
        Args:
            config: Template configuration
            topic: Topic or question
            contexts: Retrieved contexts
            **kwargs: Additional template-specific arguments
            
        Returns:
            List of LLM messages
        """
        template_func = self.templates.get(config.template_type)
        if not template_func:
            raise ValueError(f"Unknown template type: {config.template_type}")
        
        return template_func(
            topic=topic,
            contexts=contexts,
            config=config,
            **kwargs
        )
    
    def get_available_templates(self) -> Dict[str, List[str]]:
        """Get all available templates grouped by category.
        
        Returns:
            Dictionary of template categories and their templates
        """
        return {
            "问答模板": [
                "结构化答案",
                "多角度分析",
                "引导式问答",
            ],
            "学科模板": [
                "物理课",
                "历史课",
                "语文课",
                "化学课",
                "生物课",
                "数学课",
            ],
            "导学案模板": [
                "标准导学案",
            ],
            "综合模板": [
                "综合教学模板",
            ],
            "课型模板": [
                "新授课",
                "复习课",
                "实验课",
                "讨论课",
            ],
            "功能模板": [
                "习题生成",
                "学习笔记",
                "知识点总结",
                "学习路径规划",
            ],
            "交互模板": [
                "苏格拉底式",
                "辩论式",
                "项目式学习",
            ],
            "个性化模板": [
                "按年级定制",
                "按学习风格定制",
            ],
        }
    
    def get_template_description(self, template_type: TemplateType) -> str:
        """Get description for a specific template type.
        
        Args:
            template_type: Template type
            
        Returns:
            Template description
        """
        descriptions = {
            # Q&A Templates
            TemplateType.QA_STRUCTURED: "结构化答案：提供清晰、有条理的答案，包含概要、详细解释、相关概念和参考来源",
            TemplateType.QA_MULTI_PERSPECTIVE: "多角度分析：从历史、科学、应用、争议和前沿等多个视角分析问题",
            TemplateType.QA_GUIDED: "引导式问答：通过提问引导深入思考，推荐相关主题和学习路径",
            
            # Subject Templates
            TemplateType.SUBJECT_PHYSICS: "物理课教案：包含实验设计、物理模型、数学推导、物理意义和常见错误",
            TemplateType.SUBJECT_HISTORY: "历史课教案：包含历史背景、关键人物、事件经过、历史意义和史料分析",
            TemplateType.SUBJECT_CHINESE: "语文课教案：包含作者背景、作品分析、主题思想、艺术特色和相关作品",
            TemplateType.SUBJECT_CHEMISTRY: "化学课教案：包含实验设计、反应机理、化学方程式和安全注意事项",
            TemplateType.SUBJECT_BIOLOGY: "生物课教案：包含实验设计、生物概念、生命现象和生态意义",
            TemplateType.SUBJECT_MATH: "数学课教案：包含概念引入、定理证明、例题讲解和练习设计",

            # Guide Templates
            TemplateType.GUIDE_MASTER: "标准导学案：自动判断学科并按学校导学案体例生成完整成稿",
            TemplateType.GUIDE_MATH: "数学导学案：对齐学校导学案体例，包含学习目标、基础部分、要点部分、拓展部分和目标检测",
            TemplateType.GUIDE_PHYSICS: "物理导学案：突出规律分析、实验情境、典型例题、拓展训练和课堂检测",
            TemplateType.GUIDE_CHEMISTRY: "化学导学案：突出概念辨析、实验现象、性质判断、拓展应用和当堂检测",
            TemplateType.GUIDE_BIOLOGY: "生物导学案：突出概念理解、生命现象分析、探究任务和分层练习",
            TemplateType.GUIDE_HISTORY: "历史导学案：突出史实梳理、材料分析、历史意义、拓展思考和目标检测",
            TemplateType.GUIDE_GEOGRAPHY: "地理导学案：突出读图分析、区域特征、综合应用和课堂检测",
            TemplateType.GUIDE_CHINESE: "语文导学案：突出字词积累、文本赏析、主旨理解、拓展阅读和目标检测",
            TemplateType.GUIDE_ENGLISH: "英语导学案：突出词汇短语、language points、阅读任务和检测练习",
            TemplateType.GUIDE_POLITICS: "政治导学案：突出观点提炼、材料分析、现实联系和分层检测",

            # Comprehensive Template
            TemplateType.COMPREHENSIVE_MASTER: "综合教学模板：融合学科、导学案、课型、练习、互动和分层设计等优势，兼顾备课与课堂使用",
            
            # Lesson Type Templates
            TemplateType.LESSON_NEW: "新授课：导入设计、新知讲解、例题示范、练习巩固和课堂小结",
            TemplateType.LESSON_REVIEW: "复习课：知识梳理、重点回顾、难点突破、综合应用和查漏补缺",
            TemplateType.LESSON_EXPERIMENT: "实验课：实验目的、原理、器材、步骤、数据记录和结果分析",
            TemplateType.LESSON_DISCUSSION: "讨论课：讨论主题、引导问题、分组建议、讨论要点和总结提升",
            
            # Function Templates
            TemplateType.FUNC_EXERCISE: "习题生成：基础题、中等题、提高题、拓展题和详细解析",
            TemplateType.FUNC_NOTES: "学习笔记：核心概念、知识结构、重点难点、典型例题和拓展延伸",
            TemplateType.FUNC_SUMMARY: "知识点总结：概念梳理、关系图解、对比分析、应用场景和学习建议",
            TemplateType.FUNC_LEARNING_PATH: "学习路径：水平评估、目标设定、路径设计、资源推荐和时间规划",
            
            # Interactive Templates
            TemplateType.INTERACTIVE_SOCRATIC: "苏格拉底式：通过提问引导思考，不直接给出答案，培养批判性思维",
            TemplateType.INTERACTIVE_DEBATE: "辩论式：提供正反观点，引导分析论证，培养辩证思维和表达能力",
            TemplateType.INTERACTIVE_PROJECT: "项目式学习：项目背景、驱动性问题、任务分解、资源清单和评价标准",
            
            # Personalized Templates
            TemplateType.PERSONALIZED_GRADE: "年级定制：根据小学、初中、高中、大学不同年级调整内容深度和表达方式",
            TemplateType.PERSONALIZED_STYLE: "学习风格定制：根据视觉型、听觉型、动觉型、读写型调整内容呈现方式",
        }
        
        return descriptions.get(template_type, "未知模板类型")


def get_template_manager() -> TemplateManager:
    """Get singleton instance of TemplateManager."""
    global _template_manager
    if '_template_manager' not in globals():
        _template_manager = TemplateManager()
    return _template_manager
