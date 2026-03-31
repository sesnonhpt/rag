"""Template manager for the active lesson-generation templates."""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json

from src.libs.llm.base_llm import Message


class TemplateType(Enum):
    """Template types used by the current lesson-generation product."""

    GUIDE_MASTER = "guide_master"
    TEACHING_DESIGN_MASTER = "teaching_design_master"
    COMPREHENSIVE_MASTER = "comprehensive_master"


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
        """Register only the active lesson templates."""
        from .guide_templates import (
            build_guide_master_prompt,
        )
        from .comprehensive_templates import (
            build_comprehensive_master_prompt,
            build_teaching_design_prompt,
        )

        self.templates[TemplateType.GUIDE_MASTER] = build_guide_master_prompt
        self.templates[TemplateType.TEACHING_DESIGN_MASTER] = build_teaching_design_prompt
        self.templates[TemplateType.COMPREHENSIVE_MASTER] = build_comprehensive_master_prompt
    
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
            "综合模版(增强版)": [
                "综合模版(增强版)",
            ],
            "教学设计模板": [
                "教学设计",
            ],
            "导学案模板": [
                "标准导学案",
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
            TemplateType.COMPREHENSIVE_MASTER: "综合模版(增强版)：融合教学设计、导学任务、课堂互动、分层训练与配图讲解，适合生成更完整的增强版成稿",
            TemplateType.TEACHING_DESIGN_MASTER: "教学设计：贴近学校真实教学设计写法，突出教学目标、重难点、教学准备、教学过程与板书设计",
            TemplateType.GUIDE_MASTER: "标准导学案：自动判断学科并按学校导学案体例生成完整成稿",
        }
        
        return descriptions.get(template_type, "未知模板类型")


def get_template_manager() -> TemplateManager:
    """Get singleton instance of TemplateManager."""
    global _template_manager
    if '_template_manager' not in globals():
        _template_manager = TemplateManager()
    return _template_manager
